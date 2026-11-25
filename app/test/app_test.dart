import 'dart:io';

import 'package:flood_rescuer/main.dart';
import 'package:flood_rescuer/src/api.dart';
import 'package:flood_rescuer/src/app_state.dart';
import 'package:flood_rescuer/src/location.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:material_ui/material_ui.dart';

import 'fakes.dart';

class FixedLocation implements LocationSource {
  @override
  Stream<LatLng> positions() => Stream.value(const LatLng(26.5, 87.2001));
}

void main() {
  late FakeServer server;
  late AppState state;

  setUp(() {
    server = FakeServer();
    state = AppState(
      store: MemoryStore(),
      blobs: MemoryBlobs(),
      dataDir: Directory.systemTemp,
      showTiles: false,
      location: FixedLocation(),
      apiFactory: (url) => Api(url, client: server.client()),
      syncEvery: const Duration(hours: 1),
    );
  });

  Future<void> start(WidgetTester tester) async {
    await state.init();
    await tester.pumpWidget(RescuerApp(state: state));
    await tester.pumpAndSettle();
  }

  Future<void> stop(WidgetTester tester) async {
    await tester.pumpWidget(const SizedBox());
    state.dispose();
  }

  testWidgets('priority list -> route with advice and verify segments', (tester) async {
    await start(tester);
    await tester.tap(find.text('Priorities'));
    await tester.pumpAndSettle();
    expect(find.text('Ward 7'), findsOneWidget);
    expect(find.textContaining('CUT OFF'), findsOneWidget);

    await tester.tap(find.text('Route').first);
    await tester.pumpAndSettle();
    expect(find.textContaining('Ward 7: 13 min by car'), findsOneWidget);
    expect(find.text('Route is advice; the team on the ground decides.'), findsOneWidget);
    expect(find.textContaining('1 segment(s) to verify'), findsOneWidget);
    await stop(tester);
  });

  testWidgets('a report made offline waits, then is sent', (tester) async {
    await start(tester);
    server.up = false;
    await state.sync();
    await tester.pumpAndSettle();
    expect(find.text('offline'), findsOneWidget);

    await tester.tap(find.byKey(const Key('report-fab')));
    await tester.pumpAndSettle();
    expect(find.text('Your GPS position. The server picks the nearest road (within 60 m).'), findsOneWidget);
    await tester.tap(find.byKey(const Key('report-road_closed')));
    await tester.pumpAndSettle();
    expect(find.textContaining('will be sent when the phone is back online'), findsOneWidget);

    await tester.tap(find.text('Reports'));
    await tester.pumpAndSettle();
    expect(find.text('1 report(s) waiting to be sent'), findsOneWidget);
    expect(find.text('waiting'), findsOneWidget);

    server.up = true;
    await tester.tap(find.text('Sync now'));
    await tester.pumpAndSettle();
    expect(find.text('All reports sent'), findsOneWidget);
    expect(find.text('sent'), findsOneWidget);
    expect(server.reports.single['report'], 'road_closed');
    await stop(tester);
  });
}
