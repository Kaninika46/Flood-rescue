import 'package:material_ui/material_ui.dart';

import '../colors.dart';
import '../models.dart';
import '../scope.dart';

/// Ranked areas from the control room (cached for offline use).
class PrioritiesScreen extends StatelessWidget {
  const PrioritiesScreen({super.key, required this.onRouted});

  /// Called after a route was requested, to switch to the map.
  final VoidCallback onRouted;

  Future<void> _routeTo(BuildContext context, Priority p) async {
    final state = AppScope.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final err = await state.routeTo(p.centre, p.name);
    if (err != null) {
      messenger.showSnackBar(SnackBar(content: Text(err), duration: const Duration(seconds: 6)));
    } else {
      onRouted();
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    if (state.priorities.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'No priority list yet. It downloads when the phone reaches the server.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    return RefreshIndicator(
      onRefresh: state.sync,
      child: ListView.separated(
        itemCount: state.priorities.length,
        separatorBuilder: (_, _) => const Divider(height: 1),
        itemBuilder: (context, i) {
          final p = state.priorities[i];
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: p.cutOff ? Colors.black : priorityColor,
              foregroundColor: Colors.white,
              child: Text('${p.rank}'),
            ),
            title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w600)),
            subtitle: Text(
              [
                '${p.exposed.round()} people exposed',
                '${p.floodedKm2.toStringAsFixed(1)} km² flooded',
                if (p.cutOff) 'CUT OFF',
                if (p.verify) 'verify',
              ].join(' · '),
            ),
            trailing: TextButton.icon(
              onPressed: () => _routeTo(context, p),
              icon: const Icon(Icons.alt_route),
              label: const Text('Route'),
            ),
          );
        },
      ),
    );
  }
}
