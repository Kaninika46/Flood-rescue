import 'dart:math' as math;

import 'package:latlong2/latlong.dart';

import 'models.dart';

/// Distance in metres from [p] to the segment [a]-[b] (local equirectangular frame).
double pointSegmentMetres(LatLng p, LatLng a, LatLng b) {
  final kx = 111320.0 * math.cos(p.latitude * math.pi / 180), ky = 110540.0;
  final ax = (a.longitude - p.longitude) * kx, ay = (a.latitude - p.latitude) * ky;
  final bx = (b.longitude - p.longitude) * kx, by = (b.latitude - p.latitude) * ky;
  final dx = bx - ax, dy = by - ay;
  final len2 = dx * dx + dy * dy;
  final t = len2 == 0 ? 0.0 : (-(ax * dx + ay * dy) / len2).clamp(0.0, 1.0);
  final x = ax + t * dx, y = ay + t * dy;
  return math.sqrt(x * x + y * y);
}

/// The road closest to [p] within [maxMetres], or null.
Road? nearestRoad(LatLng p, Iterable<Road> roads, {double maxMetres = 60}) {
  Road? best;
  var bestD = maxMetres;
  for (final r in roads) {
    for (var i = 0; i + 1 < r.points.length; i++) {
      final d = pointSegmentMetres(p, r.points[i], r.points[i + 1]);
      if (d <= bestD) {
        bestD = d;
        best = r;
      }
    }
  }
  return best;
}

/// [west, south, east, north] around [points], padded by [padDeg].
List<double> bboxOf(Iterable<LatLng> points, {double padDeg = 0.02}) {
  var w = double.infinity, s = double.infinity, e = -double.infinity, n = -double.infinity;
  for (final p in points) {
    w = math.min(w, p.longitude);
    e = math.max(e, p.longitude);
    s = math.min(s, p.latitude);
    n = math.max(n, p.latitude);
  }
  return [w - padDeg, s - padDeg, e + padDeg, n + padDeg];
}
