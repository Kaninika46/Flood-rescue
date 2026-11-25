import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_mbtiles/flutter_map_mbtiles.dart';
import 'package:latlong2/latlong.dart';
import 'package:material_ui/material_ui.dart';

import '../colors.dart';
import '../geo.dart';
import '../models.dart';
import '../scope.dart';
import 'report_sheet.dart';

final _dashed = StrokePattern.dashed(segments: const [10, 6]);
final _dashedShort = StrokePattern.dashed(segments: const [8, 6]);

/// Offline map with the assigned route, "verify" segments, flooded roads and priorities.
class MapScreen extends StatefulWidget {
  const MapScreen({super.key, required this.controller});
  final MapController controller;

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  Road? _selected;
  LatLng? _tap;
  TileProvider? _offline;
  String? _offlinePath;

  @override
  void dispose() {
    _offline?.dispose();
    super.dispose();
  }

  Widget _tiles(BuildContext context) {
    final state = AppScope.of(context);
    if (!state.showTiles) return const SizedBox.shrink();
    if (state.hasOfflineMap) {
      if (_offlinePath != state.tilesFile.path) {
        _offline?.dispose();
        _offline = MbTilesTileProvider.fromPath(path: state.tilesFile.path);
        _offlinePath = state.tilesFile.path;
      }
      return TileLayer(tileProvider: _offline!, maxNativeZoom: 15);
    }
    return TileLayer(
      urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      userAgentPackageName: 'org.floodrescue.flood_rescuer',
    );
  }

  void _onTap(LatLng p) {
    final state = AppScope.of(context);
    setState(() {
      _selected = nearestRoad(p, state.roads, maxMetres: 40);
      _tap = p;
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final route = state.route;
    final verify = route?.verifySegments.toSet() ?? const <String>{};
    final byId = {for (final r in state.roads) r.edgeId: r};
    final centre = route?.points.first ?? state.here ?? const LatLng(26.52, 87.21);
    return Stack(
      children: [
        FlutterMap(
          mapController: widget.controller,
          options: MapOptions(
            initialCenter: centre,
            initialZoom: 13,
            maxZoom: 18,
            onTap: (_, p) => _onTap(p),
            onLongPress: (_, p) => showReportSheet(
              context,
              at: p,
              where: 'Point you pressed on the map',
              road: nearestRoad(p, state.roads, maxMetres: 40),
            ),
          ),
          children: [
            _tiles(context),
            PolylineLayer(
              polylines: [
                for (final r in state.roads)
                  if (r.status != 'dry')
                    Polyline(
                      points: r.points,
                      strokeWidth: 4,
                      color: statusColors[r.status] ?? Colors.grey,
                      pattern: r.status == 'verify' || r.status == 'unknown'
                          ? _dashed
                          : const StrokePattern.solid(),
                    ),
                if (route != null) ...[
                  Polyline(points: route.points, strokeWidth: 10, color: Colors.white),
                  Polyline(points: route.points, strokeWidth: 6, color: routeColor),
                  for (final id in verify)
                    if (byId[id] != null)
                      Polyline(
                        points: byId[id]!.points,
                        strokeWidth: 8,
                        color: statusColors['verify']!,
                        pattern: _dashedShort,
                      ),
                ],
                if (_selected != null)
                  Polyline(
                    points: _selected!.points,
                    strokeWidth: 9,
                    color: Colors.cyanAccent.withValues(alpha: 0.8),
                  ),
              ],
            ),
            MarkerLayer(
              markers: [
                for (final p in state.priorities.take(10))
                  Marker(
                    point: p.centre,
                    width: 34,
                    height: 34,
                    child: _Badge(text: '${p.rank}', color: p.cutOff ? Colors.black : priorityColor),
                  ),
                for (final q in state.queue.items.take(30))
                  Marker(
                    point: LatLng(q.lat, q.lon),
                    width: 24,
                    height: 24,
                    child: Icon(reportIcons[q.kind], color: reportColors[q.kind], size: 24),
                  ),
                if (route != null)
                  Marker(
                    point: route.points.last,
                    width: 34,
                    height: 34,
                    child: const _Badge(text: 'B', color: Color(0xFF56B4E9)),
                  ),
                if (state.here != null)
                  Marker(
                    point: state.here!,
                    width: 22,
                    height: 22,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.blue,
                        shape: BoxShape.circle,
                        border: Border.all(color: Colors.white, width: 3),
                      ),
                    ),
                  ),
              ],
            ),
            SimpleAttributionWidget(
              source: Text(
                state.hasOfflineMap
                    ? 'Offline map · OpenStreetMap contributors'
                    : 'OpenStreetMap contributors',
              ),
            ),
          ],
        ),
        if (route != null) Positioned(left: 8, right: 8, top: 8, child: _RouteCard(route: route)),
        if (_selected != null)
          Positioned(
            left: 8,
            right: 8,
            bottom: 88,
            child: Card(
              child: ListTile(
                title: Text('Road ${_selected!.edgeId}'),
                subtitle: Text(
                  [
                    'status ${_selected!.status}',
                    if (_selected!.depthM != null) 'depth ${_selected!.depthM!.toStringAsFixed(2)} m',
                    if (_selected!.bridge) 'bridge',
                  ].join(' · '),
                ),
                trailing: FilledButton(
                  onPressed: () =>
                      showReportSheet(context, at: _tap!, where: 'Selected road', road: _selected),
                  child: const Text('Report'),
                ),
                onLongPress: () => setState(() => _selected = null),
              ),
            ),
          ),
        Positioned(
          right: 16,
          bottom: 16,
          child: FloatingActionButton.extended(
            key: const Key('report-fab'),
            onPressed: () {
              final here = state.here;
              final at = here ?? widget.controller.camera.center;
              showReportSheet(
                context,
                at: at,
                where: here != null ? 'Your GPS position' : 'Map centre (no GPS fix)',
              );
            },
            icon: const Icon(Icons.campaign),
            label: const Text('Report road'),
          ),
        ),
      ],
    );
  }
}

class _Badge extends StatelessWidget {
  const _Badge({required this.text, required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
    alignment: Alignment.center,
    decoration: BoxDecoration(
      color: color,
      shape: BoxShape.circle,
      border: Border.all(color: Colors.white, width: 2.5),
      boxShadow: const [BoxShadow(blurRadius: 3, color: Colors.black38)],
    ),
    child: Text(
      text,
      style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
    ),
  );
}

class _RouteCard extends StatelessWidget {
  const _RouteCard({required this.route});
  final RouteInfo route;

  @override
  Widget build(BuildContext context) {
    final n = route.verifySegments.length;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              '${route.destination ?? 'Route'}: ${route.minutes.toStringAsFixed(0)} min by ${route.vehicle.name}',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            if (n > 0)
              Text(
                '$n segment(s) to verify on the ground (orange dashes)',
                style: const TextStyle(color: Color(0xFF8A5A00), fontWeight: FontWeight.w600),
              ),
            const SizedBox(height: 4),
            Text(route.advice, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}
