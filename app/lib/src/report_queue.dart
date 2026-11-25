import 'dart:convert';

import 'api.dart';
import 'models.dart';

/// Where the queue is kept (SharedPreferences in the app, a map in tests).
abstract class KeyValueStore {
  String? getString(String key);
  Future<void> setString(String key, String value);
}

/// Larger cached documents (road statuses): files in the app, memory in tests.
abstract class BlobStore {
  Future<String?> read(String name);
  Future<void> write(String name, String data);
}

/// Field reports are queued on the phone and sent when the server can be reached.
///
/// Reports are sent oldest first. A network failure stops the flush and keeps the rest queued;
/// a server rejection (for example no road near the position) marks that report "rejected" with
/// the reason, so the rescuer sees it, and the flush goes on.
class ReportQueue {
  ReportQueue(this._store) {
    final raw = _store.getString(_key);
    if (raw != null) {
      items.addAll((jsonDecode(raw) as List).cast<Map<String, dynamic>>().map(QueuedReport.fromJson));
    }
  }

  static const _key = 'report_queue_v1';
  final KeyValueStore _store;
  final List<QueuedReport> items = [];
  bool _flushing = false;

  int get pending => items.where((r) => r.state == ReportState.queued).length;

  Future<void> _save() => _store.setString(_key, jsonEncode([for (final r in items) r.toJson()]));

  Future<void> add(QueuedReport r) async {
    items.insert(0, r); // newest first in the list
    await _save();
  }

  /// Sends queued reports; returns how many were sent.
  Future<int> flush(Api api) async {
    if (_flushing) return 0;
    _flushing = true;
    var sent = 0;
    try {
      for (final r in items.reversed.where((r) => r.state == ReportState.queued).toList()) {
        try {
          final edge = await api.report(r);
          r
            ..state = ReportState.sent
            ..message = 'road $edge';
          sent++;
        } on ApiRejected catch (e) {
          r
            ..state = ReportState.rejected
            ..message = e.detail;
        } on ApiUnreachable {
          break;
        } finally {
          await _save();
        }
      }
    } finally {
      _flushing = false;
    }
    return sent;
  }

  /// Removes sent reports older than [age] (keeps the list short).
  Future<void> prune({Duration age = const Duration(days: 2)}) async {
    final cutoff = DateTime.now().subtract(age);
    items.removeWhere((r) => r.state == ReportState.sent && r.createdAt.isBefore(cutoff));
    await _save();
  }
}
