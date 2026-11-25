import 'dart:convert';

import 'package:flood_rescuer/src/report_queue.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

class MemoryStore implements KeyValueStore {
  final Map<String, String> data = {};
  @override
  String? getString(String key) => data[key];
  @override
  Future<void> setString(String key, String value) async => data[key] = value;
}

class MemoryBlobs implements BlobStore {
  final Map<String, String> data = {};
  @override
  Future<String?> read(String name) async => data[name];
  @override
  Future<void> write(String name, String value) async => data[name] = value;
}

/// A toy backend: two roads, two priorities, a route with one "verify" segment.
class FakeServer {
  bool up = true;
  final List<Map<String, dynamic>> reports = [];
  String vehicleWithoutRoute = 'boat';

  static Map<String, dynamic> roads() => {
    'type': 'FeatureCollection',
    'features': [
      _road('A_B_0', 'dry', [
        [87.20, 26.50],
        [87.21, 26.50],
      ]),
      _road('B_C_0', 'verify', [
        [87.21, 26.50],
        [87.22, 26.50],
      ]),
    ],
  };

  static Map<String, dynamic> _road(String id, String status, List<List<double>> c) => {
    'type': 'Feature',
    'geometry': {'type': 'LineString', 'coordinates': c},
    'properties': {
      'edge_id': id,
      'status': status,
      'confidence': 0.4,
      'max_depth_m': null,
      'source': 'level1',
      'bridge': false,
    },
  };

  static Map<String, dynamic> priorities() => {
    'type': 'FeatureCollection',
    'features': [
      for (final (i, name, cut) in [(1, 'Ward 7', true), (2, 'Ward 3', false)])
        {
          'type': 'Feature',
          'geometry': {
            'type': 'MultiPolygon',
            'coordinates': [
              [
                [
                  [87.2, 26.5],
                  [87.3, 26.5],
                  [87.3, 26.6],
                  [87.2, 26.6],
                  [87.2, 26.5],
                ],
              ],
            ],
          },
          'properties': {
            'id': i,
            'priority': i,
            'area_name': name,
            'exposed_people': 1200.0 / i,
            'flooded_km2': 3.5,
            'cut_off': cut,
            'verify': false,
            'mean_confidence': 0.8,
          },
        },
    ],
  };

  MockClient client() => MockClient((req) async {
    if (!up) throw http.ClientException('network down');
    final path = req.url.path;
    if (path == '/health') return _json({'status': 'ok'});
    if (path == '/priorities') return _json(priorities());
    if (path == '/roads/status') return _json(roads());
    if (path == '/routes') {
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      if (body['vehicle'] == vehicleWithoutRoute) {
        return _json({'detail': 'no route. Route is advice; the team on the ground decides.'}, 404);
      }
      return _json({
        'route_id': 1,
        'type': 'FeatureCollection',
        'features': [
          {
            'type': 'Feature',
            'geometry': {
              'type': 'LineString',
              'coordinates': [
                [87.20, 26.50],
                [87.21, 26.50],
                [87.22, 26.50],
              ],
            },
            'properties': {
              'edge_ids': ['A_B_0', 'B_C_0'],
              'advice': 'Route is advice; the team on the ground decides.',
            },
          },
        ],
        'advice': 'Route is advice; the team on the ground decides.',
        'verify_segments': ['B_C_0'],
        'total_time_min': 12.5,
      });
    }
    if (path == '/field_reports' && req.method == 'POST') {
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      if ((body['lat'] as num) > 30) {
        return _json({'detail': 'no road within 60 m of the report position'}, 404);
      }
      reports.add(body);
      return _json({
        'id': reports.length,
        'edge_id': body['edge_id'] ?? 'A_B_0',
        'report': body['report'],
      }, 201);
    }
    return _json({'detail': 'not found'}, 404);
  });

  static http.Response _json(Object body, [int status = 200]) =>
      http.Response(jsonEncode(body), status, headers: {'content-type': 'application/json'});
}
