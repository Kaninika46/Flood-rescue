import 'package:flood_rescuer/src/api.dart';
import 'package:flood_rescuer/src/app_state.dart';
import 'package:flood_rescuer/src/geo.dart';
import 'package:flood_rescuer/src/models.dart';
import 'package:flood_rescuer/src/report_queue.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';

import 'fakes.dart';

QueuedReport report(String id, {double lat = 26.5, ReportKind kind = ReportKind.roadClosed}) =>
    QueuedReport(localId: id, kind: kind, lat: lat, lon: 87.205, createdAt: DateTime(2024, 9, 28, 10));

void main() {
  group('Api', () {
    test('route keeps the advice text and the verify segments', () async {
      final server = FakeServer();
      final api = Api('http://server:8000/', client: server.client());
      final raw = await api.route(Vehicle.car, const LatLng(26.5, 87.2), const LatLng(26.5, 87.22));
      final r = RouteInfo.fromJson(raw, Vehicle.car, 'Ward 7');
      expect(r.advice, routeAdvice);
      expect(r.verifySegments, ['B_C_0']);
      expect(r.points.first, const LatLng(26.5, 87.2));
      expect(RouteInfo.fromCache(r.toCache(raw)).minutes, 12.5);
    });

    test('no route is a rejection, a dead network is "unreachable"', () async {
      final server = FakeServer();
      final api = Api('http://server:8000', client: server.client());
      await expectLater(
        api.route(Vehicle.boat, const LatLng(26.5, 87.2), const LatLng(26.5, 87.22)),
        throwsA(isA<ApiRejected>().having((e) => e.status, 'status', 404)),
      );
      server.up = false;
      await expectLater(api.priorities(), throwsA(isA<ApiUnreachable>()));
    });
  });

  group('ReportQueue', () {
    test('reports wait offline, then go out oldest first', () async {
      final store = MemoryStore();
      final server = FakeServer()..up = false;
      final api = Api('http://server:8000', client: server.client());
      final q = ReportQueue(store);
      await q.add(report('1', kind: ReportKind.roadClosed));
      await q.add(report('2', kind: ReportKind.bridgeGone));
      expect(await q.flush(api), 0);
      expect(q.pending, 2);

      // survives an app restart
      final again = ReportQueue(store);
      expect(again.pending, 2);

      server.up = true;
      expect(await again.flush(api), 2);
      expect(again.pending, 0);
      expect(server.reports.map((r) => r['report']), ['road_closed', 'bridge_gone']);
      expect(again.items.every((r) => r.state == ReportState.sent), isTrue);
      expect(server.reports.first.containsKey('edge_id'), isFalse); // GPS only: server snaps
    });

    test('a refused report is marked and does not block the others', () async {
      final server = FakeServer();
      final api = Api('http://server:8000', client: server.client());
      final q = ReportQueue(MemoryStore());
      await q.add(report('far', lat: 40)); // no road near: refused by the server
      await q.add(report('ok'));
      expect(await q.flush(api), 1);
      final far = q.items.firstWhere((r) => r.localId == 'far');
      expect(far.state, ReportState.rejected);
      expect(far.message, contains('no road within'));
    });

    test('reports carry the team token; a missing or wrong token keeps them queued', () async {
      final server = FakeServer()..requiredToken = 'team-secret';
      final q = ReportQueue(MemoryStore());
      await q.add(report('1'));
      await q.add(report('2'));

      final noToken = Api('http://server:8000', client: server.client());
      expect(await q.flush(noToken), 0);
      expect(q.pending, 2); // not rejected: fixed in Settings, then sent
      expect(q.items.last.message, contains('team token'));
      expect(server.reportAuth, [null]); // stopped after the first 401

      final wrong = Api('http://server:8000', token: 'nope', client: server.client());
      expect(await q.flush(wrong), 0);
      expect(q.pending, 2);

      final right = Api('http://server:8000', token: 'team-secret', client: server.client());
      expect(await q.flush(right), 2);
      expect(q.pending, 0);
      expect(server.reportAuth.last, 'Bearer team-secret');
    });

    for (final status in [429, 500, 503]) {
      test('HTTP $status keeps reports queued for a later try', () async {
        final server = FakeServer()..reportStatus = status;
        final api = Api('http://server:8000', client: server.client());
        final q = ReportQueue(MemoryStore());
        await q.add(report('1'));
        await q.add(report('2'));
        expect(await q.flush(api), 0);
        expect(q.pending, 2);
        expect(q.items.every((r) => r.state == ReportState.queued), isTrue);
        server.reportStatus = null;
        expect(await q.flush(api), 2);
      });
    }

    test('the token is sent with reports only, not with reads or routes', () async {
      final server = FakeServer();
      final api = Api('http://server:8000', token: 'team-secret', client: server.client());
      await api.priorities();
      await api.route(Vehicle.car, const LatLng(26.5, 87.2), const LatLng(26.5, 87.22));
      expect(server.reportAuth, isEmpty);
      await api.report(report('1'));
      expect(server.reportAuth, ['Bearer team-secret']);
    });
  });

  test('settings keep the team token', () {
    const s = Settings(serverUrl: 'http://x:8000', team: 'Boat 2', teamToken: 't');
    expect(Settings.fromJson(s.toJson()).teamToken, 't');
    expect(Settings.fromJson({'serverUrl': 'http://x:8000'}).teamToken, ''); // older settings
  });

  test('nearest road within the distance limit', () {
    final roads = Road.listFromGeoJson(FakeServer.roads());
    expect(nearestRoad(const LatLng(26.5001, 87.205), roads)?.edgeId, 'A_B_0'); // ~11 m
    expect(nearestRoad(const LatLng(26.51, 87.205), roads), isNull); // ~1.1 km
    final p = Priority.listFromGeoJson(FakeServer.priorities()).first;
    expect(p.centre.latitude, closeTo(26.55, 1e-9));
    expect(p.cutOff, isTrue);
  });
}
