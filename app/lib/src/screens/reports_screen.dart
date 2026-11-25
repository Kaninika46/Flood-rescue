import 'package:material_ui/material_ui.dart';

import '../colors.dart';
import '../models.dart';
import '../scope.dart';

/// The offline queue: what was sent, what is waiting, what the server refused.
class ReportsScreen extends StatelessWidget {
  const ReportsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final items = state.queue.items;
    final last = state.lastSync;
    return Column(
      children: [
        Material(
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          child: ListTile(
            leading: Icon(state.online ? Icons.cloud_done : Icons.cloud_off),
            title: Text(
              state.queue.pending == 0
                  ? 'All reports sent'
                  : '${state.queue.pending} report(s) waiting to be sent',
            ),
            subtitle: Text(last == null ? 'Never synced' : 'Last sync ${_time(last)}'),
            trailing: FilledButton(
              onPressed: state.busy ? null : state.sync,
              child: Text(state.busy ? 'Syncing…' : 'Sync now'),
            ),
          ),
        ),
        Expanded(
          child: items.isEmpty
              ? const Center(child: Text('No reports yet. Use "Report road" on the map.'))
              : ListView.separated(
                  itemCount: items.length,
                  separatorBuilder: (_, _) => const Divider(height: 1),
                  itemBuilder: (_, i) {
                    final r = items[i];
                    return ListTile(
                      leading: Icon(reportIcons[r.kind], color: reportColors[r.kind]),
                      title: Text(r.kind.label),
                      subtitle: Text(
                        [
                          _time(r.createdAt),
                          '${r.lat.toStringAsFixed(4)}, ${r.lon.toStringAsFixed(4)}',
                          if (r.message != null) r.message!,
                        ].join(' · '),
                      ),
                      trailing: _StateChip(state: r.state),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

String _time(DateTime t) =>
    '${t.day}/${t.month} ${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

class _StateChip extends StatelessWidget {
  const _StateChip({required this.state});
  final ReportState state;

  @override
  Widget build(BuildContext context) {
    final (label, color) = switch (state) {
      ReportState.queued => ('waiting', const Color(0xFFE69F00)),
      ReportState.sent => ('sent', const Color(0xFF009E73)),
      ReportState.rejected => ('refused', const Color(0xFFD55E00)),
    };
    return Chip(
      label: Text(
        label,
        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
      ),
      backgroundColor: color,
      side: BorderSide.none,
    );
  }
}
