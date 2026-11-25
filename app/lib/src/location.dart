import 'dart:async';

import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

/// Where the phone is (GPS in the app, a fixed point in tests).
abstract class LocationSource {
  Stream<LatLng> positions();
}

class GpsLocation implements LocationSource {
  @override
  Stream<LatLng> positions() async* {
    if (!await Geolocator.isLocationServiceEnabled()) return;
    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) perm = await Geolocator.requestPermission();
    if (perm == LocationPermission.denied || perm == LocationPermission.deniedForever) return;
    yield* Geolocator.getPositionStream(
      locationSettings: const LocationSettings(accuracy: LocationAccuracy.high, distanceFilter: 15),
    ).map((p) => LatLng(p.latitude, p.longitude));
  }
}

class NoLocation implements LocationSource {
  @override
  Stream<LatLng> positions() => const Stream.empty();
}
