import 'dart:io';

import 'package:flutter_map/flutter_map.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'src/app_state.dart';
import 'src/location.dart';
import 'src/report_queue.dart';
import 'src/scope.dart';
import 'src/screens/map_screen.dart';
import 'src/screens/priorities_screen.dart';
import 'src/screens/reports_screen.dart';
import 'src/screens/settings_screen.dart';

/// SharedPreferences behind the [KeyValueStore] interface.
class PrefsStore implements KeyValueStore {
  PrefsStore(this._prefs);
  final SharedPreferences _prefs;
  @override
  String? getString(String key) => _prefs.getString(key);
  @override
  Future<void> setString(String key, String value) => _prefs.setString(key, value);
}

/// Files in the app's documents folder behind the [BlobStore] interface.
class FileBlobStore implements BlobStore {
  FileBlobStore(this.dir);
  final Directory dir;
  @override
  Future<String?> read(String name) async {
    final f = File('${dir.path}/$name');
    return f.existsSync() ? f.readAsString() : null;
  }

  @override
  Future<void> write(String name, String data) => File('${dir.path}/$name').writeAsString(data);
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final prefs = await SharedPreferences.getInstance();
  final dir = await getApplicationDocumentsDirectory();
  final state = AppState(
    store: PrefsStore(prefs),
    blobs: FileBlobStore(dir),
    dataDir: dir,
    location: GpsLocation(),
  );
  await state.init();
  runApp(RescuerApp(state: state));
}

class RescuerApp extends StatelessWidget {
  const RescuerApp({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) => AppScope(
    state: state,
    child: MaterialApp(
      title: 'Flood Rescue',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0F2740)),
        visualDensity: VisualDensity.comfortable,
      ),
      home: const HomeScreen(),
    ),
  );
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _tab = 0;
  final _map = MapController();

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final pending = state.queue.pending;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Flood Rescue'),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Chip(
              avatar: Icon(state.online ? Icons.wifi : Icons.wifi_off, size: 18),
              label: Text(state.online ? 'online' : 'offline'),
            ),
          ),
          IconButton(
            tooltip: 'Settings',
            icon: const Icon(Icons.settings),
            onPressed: () =>
                Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const SettingsScreen())),
          ),
        ],
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(22),
          child: Padding(
            padding: EdgeInsets.only(bottom: 4),
            child: Text(
              'EXERCISE · student project · routes are advice',
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
            ),
          ),
        ),
      ),
      body: IndexedStack(
        index: _tab,
        children: [
          MapScreen(controller: _map),
          PrioritiesScreen(onRouted: () => setState(() => _tab = 0)),
          const ReportsScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: [
          const NavigationDestination(icon: Icon(Icons.map), label: 'Map'),
          const NavigationDestination(icon: Icon(Icons.format_list_numbered), label: 'Priorities'),
          NavigationDestination(
            icon: Badge(
              isLabelVisible: pending > 0,
              label: Text('$pending'),
              child: const Icon(Icons.outbox),
            ),
            label: 'Reports',
          ),
        ],
      ),
    );
  }
}
