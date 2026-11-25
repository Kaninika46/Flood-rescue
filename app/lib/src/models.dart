import 'package:latlong2/latlong.dart';

/// The advice text every route carries (same as the backend's `ADVICE`).
const routeAdvice = 'Route is advice; the team on the ground decides.';

enum Vehicle { foot, car, truck, boat }

/// Field report types offered in the app (the backend also accepts `flooded`).
enum ReportKind {
  roadOpen('road_open', 'Road open'),
  roadClosed('road_closed', 'Road closed'),
  bridgeGone('bridge_gone', 'Bridge gone');

  const ReportKind(this.api, this.label);
  final String api;
  final String label;

  static ReportKind fromApi(String s) => values.firstWhere((k) => k.api == s);
}

List<LatLng> _line(List<dynamic> coords) => [
  for (final c in coords) LatLng((c[1] as num).toDouble(), (c[0] as num).toDouble()),
];

/// One road edge with its flood status (`GET /roads/status`).
class Road {
  Road({
    required this.edgeId,
    required this.status,
    required this.points,
    this.depthM,
    this.confidence,
    this.bridge = false,
  });

  final String edgeId;
  final String status; // dry | flooded | verify | unknown | closed
  final List<LatLng> points;
  final double? depthM;
  final double? confidence;
  final bool bridge;

  static List<Road> listFromGeoJson(Map<String, dynamic> fc) => [
    for (final f in (fc['features'] as List).cast<Map<String, dynamic>>())
      if (f['geometry'] != null && f['geometry']['type'] == 'LineString')
        Road(
          edgeId: f['properties']['edge_id'] as String,
          status: f['properties']['status'] as String,
          points: _line(f['geometry']['coordinates'] as List),
          depthM: (f['properties']['max_depth_m'] as num?)?.toDouble(),
          confidence: (f['properties']['confidence'] as num?)?.toDouble(),
          bridge: f['properties']['bridge'] == true,
        ),
  ];
}

/// A ranked area (`GET /priorities`).
class Priority {
  Priority({
    required this.id,
    required this.rank,
    required this.name,
    required this.exposed,
    required this.floodedKm2,
    required this.cutOff,
    required this.verify,
    required this.centre,
  });

  final int id;
  final int rank;
  final String name;
  final double exposed;
  final double floodedKm2;
  final bool cutOff;
  final bool verify;
  final LatLng centre;

  static List<Priority> listFromGeoJson(Map<String, dynamic> fc) => [
    for (final f in (fc['features'] as List).cast<Map<String, dynamic>>())
      Priority(
        id: f['properties']['id'] as int,
        rank: f['properties']['priority'] as int,
        name: f['properties']['area_name'] as String,
        exposed: (f['properties']['exposed_people'] as num).toDouble(),
        floodedKm2: (f['properties']['flooded_km2'] as num).toDouble(),
        cutOff: f['properties']['cut_off'] == true,
        verify: f['properties']['verify'] == true,
        centre: bboxCentre(f['geometry'] as Map<String, dynamic>),
      ),
  ];
}

/// Centre of the bounding box of a GeoJSON geometry.
LatLng bboxCentre(Map<String, dynamic> geometry) {
  var w = double.infinity, s = double.infinity, e = -double.infinity, n = -double.infinity;
  void visit(dynamic c) {
    if (c is List && c.isNotEmpty && c.first is num) {
      final x = (c[0] as num).toDouble(), y = (c[1] as num).toDouble();
      if (x < w) w = x;
      if (x > e) e = x;
      if (y < s) s = y;
      if (y > n) n = y;
    } else if (c is List) {
      c.forEach(visit);
    }
  }

  visit(geometry['coordinates']);
  return LatLng((s + n) / 2, (w + e) / 2);
}

/// A route from `POST /routes`.
class RouteInfo {
  RouteInfo({
    required this.points,
    required this.minutes,
    required this.verifySegments,
    required this.edgeIds,
    required this.advice,
    required this.vehicle,
    this.destination,
  });

  final List<LatLng> points;
  final double minutes;
  final List<String> verifySegments;
  final List<String> edgeIds;
  final String advice;
  final Vehicle vehicle;
  final String? destination;

  factory RouteInfo.fromJson(Map<String, dynamic> j, Vehicle vehicle, String? destination) {
    final f = (j['features'] as List).first as Map<String, dynamic>;
    return RouteInfo(
      points: _line(f['geometry']['coordinates'] as List),
      minutes: (j['total_time_min'] as num).toDouble(),
      verifySegments: (j['verify_segments'] as List).cast<String>(),
      edgeIds: ((f['properties'] as Map)['edge_ids'] as List? ?? const []).cast<String>(),
      advice: (j['advice'] as String?) ?? routeAdvice,
      vehicle: vehicle,
      destination: destination,
    );
  }

  /// JSON kept on the phone so the route stays visible offline.
  Map<String, dynamic> toCache(Map<String, dynamic> raw) => {
    'raw': raw,
    'vehicle': vehicle.name,
    'destination': destination,
  };

  static RouteInfo fromCache(Map<String, dynamic> c) => RouteInfo.fromJson(
    c['raw'] as Map<String, dynamic>,
    Vehicle.values.byName(c['vehicle'] as String),
    c['destination'] as String?,
  );
}

enum ReportState { queued, sent, rejected }

/// A field report waiting in (or sent from) the offline queue.
class QueuedReport {
  QueuedReport({
    required this.localId,
    required this.kind,
    required this.lat,
    required this.lon,
    required this.createdAt,
    this.edgeId,
    this.reporter,
    this.note,
    this.state = ReportState.queued,
    this.message,
  });

  final String localId;
  final ReportKind kind;
  final double lat;
  final double lon;
  final DateTime createdAt;
  final String? edgeId;
  final String? reporter;
  final String? note;
  ReportState state;
  String? message; // server answer: snapped edge id, or why it was rejected

  Map<String, dynamic> toBody() => {
    if (edgeId != null) 'edge_id': edgeId,
    'report': kind.api,
    'lon': lon,
    'lat': lat,
    if (reporter != null && reporter!.isNotEmpty) 'reporter': reporter,
    'note': [
      if (note != null && note!.isNotEmpty) note!,
      'queued ${createdAt.toIso8601String()}',
    ].join(' | '),
  };

  Map<String, dynamic> toJson() => {
    'localId': localId,
    'kind': kind.api,
    'lat': lat,
    'lon': lon,
    'createdAt': createdAt.toIso8601String(),
    'edgeId': edgeId,
    'reporter': reporter,
    'note': note,
    'state': state.name,
    'message': message,
  };

  static QueuedReport fromJson(Map<String, dynamic> j) => QueuedReport(
    localId: j['localId'] as String,
    kind: ReportKind.fromApi(j['kind'] as String),
    lat: (j['lat'] as num).toDouble(),
    lon: (j['lon'] as num).toDouble(),
    createdAt: DateTime.parse(j['createdAt'] as String),
    edgeId: j['edgeId'] as String?,
    reporter: j['reporter'] as String?,
    note: j['note'] as String?,
    state: ReportState.values.byName(j['state'] as String),
    message: j['message'] as String?,
  );
}
