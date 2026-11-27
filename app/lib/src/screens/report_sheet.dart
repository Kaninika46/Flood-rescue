import 'package:latlong2/latlong.dart';
import 'package:material_ui/material_ui.dart';

import '../colors.dart';
import '../models.dart';
import '../scope.dart';

/// Bottom sheet with the three report buttons. [at] is the GPS position or a point on the map.
Future<void> showReportSheet(BuildContext context, {required LatLng at, required String where, Road? road}) {
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    builder: (_) => ReportSheet(at: at, where: where, road: road),
  );
}

class ReportSheet extends StatefulWidget {
  const ReportSheet({super.key, required this.at, required this.where, this.road});
  final LatLng at;
  final String where;
  final Road? road;

  @override
  State<ReportSheet> createState() => _ReportSheetState();
}

class _ReportSheetState extends State<ReportSheet> {
  final _note = TextEditingController();
  bool _busy = false;

  @override
  void dispose() {
    _note.dispose();
    super.dispose();
  }

  Future<void> _send(ReportKind kind) async {
    setState(() => _busy = true);
    final state = AppScope.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final nav = Navigator.of(context);
    await state.addReport(kind, widget.at, edgeId: widget.road?.edgeId, note: _note.text.trim());
    nav.pop();
    messenger.showSnackBar(
      SnackBar(
        content: Text(
          state.online && state.queue.pending == 0
              ? '${kind.label} sent.'
              : '${kind.label} saved. It will be sent when the phone is back online.',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final road = widget.road;
    return Padding(
      padding: EdgeInsets.fromLTRB(20, 16, 20, 20 + MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Report this road', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 4),
          Text(
            road == null
                ? '${widget.where}. The server picks the nearest road (within 60 m).'
                : 'Road ${road.edgeId} (now: ${road.status}${road.bridge ? ', bridge' : ''}).',
          ),
          Text(
            '${widget.at.latitude.toStringAsFixed(5)}, ${widget.at.longitude.toStringAsFixed(5)}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
          for (final kind in ReportKind.values)
            Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: FilledButton.icon(
                key: Key('report-${kind.api}'),
                style: FilledButton.styleFrom(
                  backgroundColor: reportColors[kind],
                  minimumSize: const Size.fromHeight(56),
                  textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                ),
                onPressed: _busy ? null : () => _send(kind),
                icon: Icon(reportIcons[kind]),
                label: Text(kind.label),
              ),
            ),
          TextField(
            controller: _note,
            maxLength: 150, // the queue adds " | queued <time>" (server limit 200)
            decoration: const InputDecoration(labelText: 'Note (optional)', border: OutlineInputBorder()),
          ),
        ],
      ),
    );
  }
}
