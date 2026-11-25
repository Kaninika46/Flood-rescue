import 'package:material_ui/material_ui.dart';

import '../app_state.dart';
import '../models.dart';
import '../scope.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _url;
  late final TextEditingController _team;
  late Vehicle _vehicle;
  String? _mapMessage;
  bool _downloading = false;
  bool _ready = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_ready) return;
    final s = AppScope.of(context).settings;
    _url = TextEditingController(text: s.serverUrl);
    _team = TextEditingController(text: s.team);
    _vehicle = s.vehicle;
    _ready = true;
  }

  @override
  void dispose() {
    _url.dispose();
    _team.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'Server address',
              helperText: 'The laptop running the backend, e.g. http://192.168.1.20:8000',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _team,
            decoration: const InputDecoration(labelText: 'Team name', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<Vehicle>(
            initialValue: _vehicle,
            decoration: const InputDecoration(labelText: 'Vehicle', border: OutlineInputBorder()),
            items: const [
              DropdownMenuItem(value: Vehicle.car, child: Text('Car (max 0.15 m water)')),
              DropdownMenuItem(value: Vehicle.truck, child: Text('Truck (max 0.30 m water)')),
              DropdownMenuItem(value: Vehicle.foot, child: Text('On foot (max 0.15 m water)')),
              DropdownMenuItem(value: Vehicle.boat, child: Text('Boat (flooded roads only)')),
            ],
            onChanged: (v) => setState(() => _vehicle = v ?? _vehicle),
          ),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: () async {
              await state.saveSettings(
                Settings(serverUrl: _url.text.trim(), team: _team.text.trim(), vehicle: _vehicle),
              );
              if (context.mounted) Navigator.of(context).pop();
            },
            child: const Text('Save'),
          ),
          const Divider(height: 40),
          Text('Offline map', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Text(
            state.hasOfflineMap
                ? 'Saved on this phone. The map works without internet.'
                : 'Not on this phone yet. Download it at the base, on Wi-Fi.',
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: _downloading
                ? null
                : () async {
                    setState(() => _downloading = true);
                    final msg = await state.downloadOfflineMap();
                    if (mounted) {
                      setState(() {
                        _downloading = false;
                        _mapMessage = msg;
                      });
                    }
                  },
            icon: const Icon(Icons.download),
            label: Text(_downloading ? 'Downloading…' : 'Download offline map'),
          ),
          if (_mapMessage != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_mapMessage!)),
        ],
      ),
    );
  }
}
