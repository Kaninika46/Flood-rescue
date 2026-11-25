import 'package:material_ui/material_ui.dart';

import 'models.dart';

/// Same palette as the dashboard (dashboard/src/colors.ts).
const statusColors = <String, Color>{
  'flooded': Color(0xFF0072B2),
  'closed': Color(0xFF1A1A1A),
  'verify': Color(0xFFE69F00),
  'unknown': Color(0xFF8E44AD),
  'dry': Color(0xFF8A9099),
};
const routeColor = Color(0xFFCC79A7);
const priorityColor = Color(0xFFD55E00);

const reportColors = <ReportKind, Color>{
  ReportKind.roadOpen: Color(0xFF009E73),
  ReportKind.roadClosed: Color(0xFF1A1A1A),
  ReportKind.bridgeGone: Color(0xFFD55E00),
};

const reportIcons = <ReportKind, IconData>{
  ReportKind.roadOpen: Icons.check_circle,
  ReportKind.roadClosed: Icons.block,
  ReportKind.bridgeGone: Icons.warning_amber,
};
