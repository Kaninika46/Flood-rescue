import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:latlong2/latlong.dart';

import 'api.dart';
import 'geo.dart';
import 'location.dart';
import 'models.dart';
import 'report_queue.dart';

/// Team settings kept on the phone.
class Settings {
  const Settings({this.serverUrl = 'http://10.0.2.2:8000', this.team = '', this.vehicle = Vehicle.car});
  final String serverUrl; // 10.0.2.2 = the laptop, seen from the Android emulator
  final String team;
  final Vehicle vehicle;

  Map<String, dynamic> toJson() => {'serverUrl': serverUrl, 'team': team, 'vehicle': vehicle.name};
  static Settings fromJson(Map<String, dynamic> j) => Settings(
    serverUrl: j['serverUrl'] as String,
    team: j['team'] as String? ?? '',
    vehicle: Vehicle.values.byName(j['vehicle'] as String? ?? 'car'),
  );
}

/// Everything the screens show. Data is cached on the phone, so the last priorities, roads and
/// route stay visible offline; field reports wait in [queue] until the server is reachable.
class AppState extends ChangeNotifier {
  AppState({
    required this.store,
    required this.blobs,
    required this.dataDir,
    this.showTiles = true,
    LocationSource? location,
    Api Function(String baseUrl)? apiFactory,
    this.syncEvery = const Duration(seconds: 30),
  }) : _location = location ?? NoLocation(),
       _apiFactory = apiFactory ?? Api.new;

  final KeyValueStore store;
  final BlobStore blobs;
  final Directory dataDir;
  final bool showTiles; // false in widget tests (no map tile downloads)
  final Duration syncEvery;
  final LocationSource _location;
  final Api Function(String) _apiFactory;

  Settings settings = const Settings();
  late final ReportQueue queue = ReportQueue(store);
  List<Priority> priorities = [];
  List<Road> roads = [];
  RouteInfo? route;
  LatLng? here;
  bool online = false;
  bool busy = false;
  DateTime? lastSync;
  String? lastError;
  Timer? _timer;
  StreamSubscription<LatLng>? _gps;
  Api? _api;

  Api get api => _api ??= _apiFactory(settings.serverUrl);
  File get tilesFile => File('${dataDir.path}/koshi.mbtiles');
  bool get hasOfflineMap => tilesFile.existsSync();

  Future<void> init() async {
    final s = store.getString('settings');
    if (s != null) settings = Settings.fromJson(jsonDecode(s) as Map<String, dynamic>);
    final p = store.getString('priorities');
    if (p != null) priorities = Priority.listFromGeoJson(jsonDecode(p) as Map<String, dynamic>);
    final r = store.getString('route');
    if (r != null && r.isNotEmpty) route = RouteInfo.fromCache(jsonDecode(r) as Map<String, dynamic>);
    final cached = await blobs.read('roads.json');
    if (cached != null) roads = Road.listFromGeoJson(jsonDecode(cached) as Map<String, dynamic>);
    final t = store.getString('last_sync');
    if (t != null) lastSync = DateTime.parse(t);
    _gps = _location.positions().listen((p) {
      here = p;
      notifyListeners();
    }, onError: (_) {});
    notifyListeners();
    unawaited(sync());
    _timer = Timer.periodic(syncEvery, (_) => unawaited(sync()));
  }

  @override
  void dispose() {
    _timer?.cancel();
    _gps?.cancel();
    super.dispose();
  }

  Future<void> saveSettings(Settings s) async {
    settings = s;
    _api = null;
    await store.setString('settings', jsonEncode(s.toJson()));
    notifyListeners();
    unawaited(sync());
  }

  /// Send queued reports, then refresh priorities and roads. Never throws.
  Future<void> sync() async {
    if (busy) return;
    busy = true;
    notifyListeners();
    try {
      await queue.flush(api);
      final p = await api.priorities();
      priorities = Priority.listFromGeoJson(p);
      await store.setString('priorities', jsonEncode(p));
      final area = route != null
          ? bboxOf(route!.points, padDeg: 0.03)
          : here != null
          ? bboxOf([here!], padDeg: 0.05)
          : null;
      final r = await api.roads(bbox: area);
      roads = Road.listFromGeoJson(r);
      await blobs.write('roads.json', jsonEncode(r));
      online = true;
      lastError = null;
      lastSync = DateTime.now();
      await store.setString('last_sync', lastSync!.toIso8601String());
    } on ApiUnreachable {
      online = false;
    } on ApiRejected catch (e) {
      online = true;
      lastError = e.detail;
    } finally {
      busy = false;
      notifyListeners();
    }
  }

  /// Ask the server for a route from here (or [from]) to [dest]; returns an error text or null.
  Future<String?> routeTo(LatLng dest, String name, {LatLng? from}) async {
    final start = from ?? here;
    if (start == null) return 'No GPS position yet. Long-press the map to set your start.';
    try {
      final raw = await api.route(settings.vehicle, start, dest);
      route = RouteInfo.fromJson(raw, settings.vehicle, name);
      await store.setString('route', jsonEncode(route!.toCache(raw)));
      online = true;
      notifyListeners();
      unawaited(sync()); // roads around the new route
      return null;
    } on ApiUnreachable {
      online = false;
      notifyListeners();
      return 'Offline: cannot get a new route. The last route stays on the map.';
    } on ApiRejected catch (e) {
      return e.status == 404 ? 'No safe route for a ${settings.vehicle.name}. $routeAdvice' : e.detail;
    }
  }

  Future<void> clearRoute() async {
    route = null;
    await store.setString('route', '');
    notifyListeners();
  }

  /// Queue a field report and try to send it at once.
  Future<void> addReport(ReportKind kind, LatLng at, {String? edgeId, String? note}) async {
    await queue.add(
      QueuedReport(
        localId: '${DateTime.now().microsecondsSinceEpoch}',
        kind: kind,
        lat: at.latitude,
        lon: at.longitude,
        createdAt: DateTime.now(),
        edgeId: edgeId,
        reporter: settings.team.isEmpty ? 'rescuer app' : 'app: ${settings.team}',
        note: note,
      ),
    );
    notifyListeners();
    await sync();
    if (online && route != null) {
      // the report may change the route: ask again from the same start
      await routeTo(
        route!.points.last,
        route!.destination ?? 'destination',
        from: here ?? route!.points.first,
      );
    }
  }

  Future<String> downloadOfflineMap() async {
    try {
      final n = await api.downloadTiles(tilesFile);
      notifyListeners();
      return 'Offline map saved (${(n / 1e6).toStringAsFixed(1)} MB).';
    } on ApiUnreachable {
      return 'Server not reachable. Connect to the base Wi-Fi and try again.';
    } on ApiRejected catch (e) {
      return e.detail;
    }
  }
}
