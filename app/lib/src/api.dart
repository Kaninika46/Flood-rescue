import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';

import 'models.dart';

/// The server answered with an error (4xx/5xx): retrying the same request will not help.
class ApiRejected implements Exception {
  ApiRejected(this.status, this.detail);
  final int status;
  final String detail;
  @override
  String toString() => detail;
}

/// The server could not be reached (offline, timeout): try again later.
class ApiUnreachable implements Exception {
  ApiUnreachable(this.cause);
  final Object cause;
  @override
  String toString() => 'server not reachable';
}

/// Client for the FastAPI backend.
class Api {
  Api(this.baseUrl, {http.Client? client, this.timeout = const Duration(seconds: 12)})
    : _client = client ?? http.Client();

  final String baseUrl;
  final Duration timeout;
  final http.Client _client;

  Uri _u(String path, [Map<String, String>? q]) {
    final base = baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;
    return Uri.parse('$base$path').replace(queryParameters: q);
  }

  Future<dynamic> _send(Future<http.Response> Function() call) async {
    final http.Response r;
    try {
      r = await call().timeout(timeout);
    } on SocketException catch (e) {
      throw ApiUnreachable(e);
    } on http.ClientException catch (e) {
      throw ApiUnreachable(e);
    } on HandshakeException catch (e) {
      throw ApiUnreachable(e);
    } on TimeoutException catch (e) {
      throw ApiUnreachable(e);
    }
    final body = r.body.isEmpty ? null : jsonDecode(utf8.decode(r.bodyBytes));
    if (r.statusCode >= 400) {
      final detail = body is Map && body['detail'] != null ? '${body['detail']}' : 'HTTP ${r.statusCode}';
      throw ApiRejected(r.statusCode, detail);
    }
    return body;
  }

  Future<dynamic> _get(String path, [Map<String, String>? q]) => _send(() => _client.get(_u(path, q)));

  Future<dynamic> _post(String path, Map<String, dynamic> body) => _send(
    () => _client.post(_u(path), headers: {'content-type': 'application/json'}, body: jsonEncode(body)),
  );

  Future<bool> health() async => (await _get('/health'))['status'] == 'ok';

  Future<Map<String, dynamic>> priorities() async => (await _get('/priorities')) as Map<String, dynamic>;

  /// Road statuses, optionally only inside [bbox] = (west, south, east, north).
  Future<Map<String, dynamic>> roads({List<double>? bbox}) async => (await _get(
    '/roads/status',
    bbox == null ? null : {'bbox': bbox.map((x) => x.toStringAsFixed(5)).join(',')},
  )) as Map<String, dynamic>;

  /// Raw route JSON (cache it) for a vehicle between two points.
  Future<Map<String, dynamic>> route(Vehicle vehicle, LatLng start, LatLng end) async =>
      (await _post('/routes', {
        'vehicle': vehicle.name,
        'start': [start.longitude, start.latitude],
        'end': [end.longitude, end.latitude],
      })) as Map<String, dynamic>;

  /// Sends one report; returns the edge id the server used.
  Future<String> report(QueuedReport r) async =>
      ((await _post('/field_reports', r.toBody())) as Map<String, dynamic>)['edge_id'] as String;

  /// Downloads the offline map (MBTiles) to [file].
  Future<int> downloadTiles(File file) async {
    final http.StreamedResponse r;
    try {
      r = await _client.send(http.Request('GET', _u('/offline/tiles.mbtiles')));
    } on Exception catch (e) {
      throw ApiUnreachable(e);
    }
    if (r.statusCode != 200) throw ApiRejected(r.statusCode, 'no offline map on the server');
    final tmp = File('${file.path}.part');
    final sink = tmp.openWrite();
    var n = 0;
    await for (final chunk in r.stream) {
      sink.add(chunk);
      n += chunk.length;
    }
    await sink.close();
    await tmp.rename(file.path);
    return n;
  }
}
