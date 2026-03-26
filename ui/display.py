"""
Pygame observer UI for the rail strategy game.

Displays:
  - Rail network map (stations as circles, edges as lines)
  - Chip counts and station control colours
  - Team positions and route lines
  - Challenge markers
  - Score panel (coins, controlled stations, day/time)
  - Pause / Play / Speed controls

Human player interaction (when human_team_id is set):
  Phase 1 — departure selection: destination badges are drawn on the map.
             Click a badge to select that trip.
  Phase 2 — chip configuration: a per-stop +/- list appears in the side panel.
             Adjust coins per stop, then click CONFIRM to board, or CANCEL to go back.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import pygame

from engine.game_state import GameState, Station
from engine.rail_network import Departure, RailNetwork
from engine.rules import compute_route_chip_cost, count_controlled_stations
from engine.clock import sim_minute_to_wall_clock

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG          = (15,  15,  20)
NEUTRAL     = (100, 100, 110)
TEAM_A      = (60,  130, 220)   # blue
TEAM_B      = (220,  70,  60)   # red
CONTESTED   = (160,  80, 200)   # purple
EDGE_COL    = (40,   40,  50)
CHALLENGE_C = (255, 210,  50)   # gold
TEXT_COL    = (220, 220, 230)
PANEL_BG    = (25,  25,  35)
BTN_ACTIVE  = (60, 160,  80)
BTN_IDLE    = (50,  50,  65)
BTN_TEXT    = (200, 200, 200)
ROUTE_A     = (*TEAM_A[:3], 180)
ROUTE_B     = (*TEAM_B[:3], 180)

# Human overlay colours
OVERLAY_BG         = (20,  20,  30, 200)
OVERLAY_BORDER_OK  = (80,  200,  80)
OVERLAY_BORDER_NO  = (200,  80,  80)
OVERLAY_SELECTED   = (255, 220,  60)   # gold — selected departure badge
OVERLAY_TEXT       = (240, 240, 240)
CHIP_BTN_ACTIVE    = (100, 200, 100)
CHIP_BTN_IDLE      = (50,   55,  70)
SKIP_BTN_COL       = (160, 120,  40)
CONFIRM_BTN_COL    = (50,  160,  80)
CANCEL_BTN_COL     = (130,  50,  50)
HUMAN_PULSE        = (255, 255, 100)   # bright ring on human's station

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
MAP_MARGIN    = 30
PANEL_W       = 280
MIN_STATION_R = 7
MAX_STATION_R = 22
LOG_LINES     = 14
# Max intermediate stops shown in the per-stop chip panel
MAX_STOP_ROWS = 10
STOP_ROW_H    = 20   # px per stop row in panel


class Display:
    """
    Manages the Pygame window and renders the game state.

    Call display.handle_events() + display.draw(state, net, log) each frame.
    """

    def __init__(
        self,
        config: dict,
        rail_network: RailNetwork,
        starting_station_id: str,
        human_team_id: Optional[str] = None,
    ):
        self.config = config
        self.net = rail_network
        self.starting_station_id = starting_station_id
        self.speed_options: List[float] = config["simulation"]["speed_options"]
        self.human_team_id: Optional[str] = human_team_id
        self._max_chips: int = config["game"].get("max_chip_differential", 5)

        pygame.init()
        info = pygame.display.Info()
        self.W = info.current_w - 20
        self.H = info.current_h - 60
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Rail Strategy Game")

        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_md  = pygame.font.SysFont("monospace", 13)
        self.font_lg  = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_xl  = pygame.font.SysFont("monospace", 20, bold=True)

        self.map_rect = pygame.Rect(
            MAP_MARGIN, MAP_MARGIN,
            self.W - PANEL_W - MAP_MARGIN * 2,
            self.H - MAP_MARGIN * 2,
        )

        # Pre-compute station pixel positions and edge list
        self._station_px: Dict[str, Tuple[int, int]] = {}
        self._edges: Set[Tuple[str, str]] = set()
        self._build_layout()

        # Control state
        self.is_paused: bool = False
        self.speed_idx: int  = 0   # index into speed_options
        self._buttons: List[dict] = []
        self._skip_button: Optional[dict] = None
        self._build_buttons()

        # Human click tracking (set during handle_events, consumed by handle_human_click)
        self._last_click_pos: Optional[Tuple[int, int]] = None

        # Human player per-stop chip configuration state
        # Phase 1: _selected_dep_idx is None  → click a destination badge to select
        # Phase 2: _selected_dep_idx is set   → configure chips per stop, then confirm
        self._selected_dep_idx: Optional[int] = None
        self._pending_chips: List[int] = []          # coins per stop (phase 2)
        self._stop_chip_btns: List[dict] = []         # [{minus_rect, plus_rect, stop_idx}, ...]
        self._confirm_btn: Optional[dict] = None
        self._cancel_btn: Optional[dict] = None

        self._log_surface = pygame.Surface((PANEL_W - 20, LOG_LINES * 16))

    # ------------------------------------------------------------------
    # Layout pre-computation
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        stations = list(self.net.stations.values())
        if not stations:
            return

        lats = [s.lat for s in stations]
        lons = [s.lon for s in stations]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_span = max_lat - min_lat or 1
        lon_span = max_lon - min_lon or 1

        for s in stations:
            # Longitude → x, Latitude → y (flip y so north is up)
            nx = (s.lon - min_lon) / lon_span
            ny = 1.0 - (s.lat - min_lat) / lat_span
            px = int(self.map_rect.left + nx * self.map_rect.width)
            py = int(self.map_rect.top  + ny * self.map_rect.height)
            self._station_px[s.id] = (px, py)

        # Build edges from schedules (pairs of consecutive stops on any trip)
        seen_trips: Dict[str, List[str]] = {}
        for sid, deps in self.net.schedules.items():
            for dep in deps[:3]:   # sample a few trips per station
                tid = dep.trip_id
                if tid not in seen_trips:
                    seen_trips[tid] = [sid] + dep.intermediate_stops
        for stops in seen_trips.values():
            for a, b in zip(stops, stops[1:]):
                if a != b:
                    key = (min(a, b), max(a, b))
                    self._edges.add(key)

    def _build_buttons(self) -> None:
        bx = self.W - PANEL_W + 10
        by = self.H - 130
        bw, bh = (PANEL_W - 20) // 2 - 5, 32

        self._buttons = [
            {"label": "|| PAUSE", "rect": pygame.Rect(bx, by, bw, bh),            "action": "pause"},
            {"label": "> PLAY",   "rect": pygame.Rect(bx + bw + 10, by, bw, bh),  "action": "play"},
            {"label": "SPEED +",  "rect": pygame.Rect(bx, by + bh + 8, bw, bh),   "action": "speed_up"},
            {"label": "SPEED -",  "rect": pygame.Rect(bx + bw + 10, by + bh + 8, bw, bh), "action": "speed_down"},
        ]

        if self.human_team_id is not None:
            # Skip button sits just above the speed buttons
            skip_h = 26
            self._skip_button = {
                "label": "SKIP 30 MIN",
                "rect": pygame.Rect(bx, by - skip_h - 8, PANEL_W - 20, skip_h),
            }

    def _select_departure(self, dep_idx: int, departures: List[Departure]) -> None:
        """Enter phase 2: build per-stop +/- button rects for the chosen departure."""
        self._selected_dep_idx = dep_idx
        dep = departures[dep_idx]
        n = len(dep.intermediate_stops)
        # Copy existing chips_per_stop (may already have values from previous edit)
        self._pending_chips = list(dep.chips_per_stop[:n])
        if len(self._pending_chips) < n:
            self._pending_chips += [0] * (n - len(self._pending_chips))

        # Build button rects anchored above the skip button
        skip_top = self._skip_button["rect"].top if self._skip_button else (self.H - 170)
        btn_h = 26
        confirm_y = skip_top - btn_h - 8
        visible = min(n, MAX_STOP_ROWS)
        rows_h = visible * STOP_ROW_H
        rows_y = confirm_y - rows_h - 24  # 24px for section header

        bx = self.W - PANEL_W + 10
        btn_w_small = 22

        self._stop_chip_btns = []
        for row in range(visible):
            row_y = rows_y + row * STOP_ROW_H
            minus_rect = pygame.Rect(bx + 110, row_y, btn_w_small, STOP_ROW_H - 2)
            plus_rect  = pygame.Rect(bx + 110 + btn_w_small + 20, row_y, btn_w_small, STOP_ROW_H - 2)
            self._stop_chip_btns.append({
                "stop_idx": row,
                "minus_rect": minus_rect,
                "plus_rect": plus_rect,
                "row_y": row_y,
            })

        # Confirm / cancel buttons
        half = (PANEL_W - 20) // 2 - 4
        self._confirm_btn = {"label": "CONFIRM", "rect": pygame.Rect(bx, confirm_y, half, btn_h)}
        self._cancel_btn  = {"label": "CANCEL",  "rect": pygame.Rect(bx + half + 8, confirm_y, half, btn_h)}

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self) -> Optional[str]:
        """
        Process pygame events. Returns:
          "quit"       — user closed the window
          "pause"      — toggle pause
          "speed_up"   — increase speed
          "speed_down" — decrease speed
          None         — no relevant event

        Mouse clicks that don't match control buttons are stored in
        self._last_click_pos for handle_human_click() to process.
        """
        self._last_click_pos = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return "pause"
                if event.key == pygame.K_UP:
                    return "speed_up"
                if event.key == pygame.K_DOWN:
                    return "speed_down"
                if event.key == pygame.K_ESCAPE and self._selected_dep_idx is not None:
                    # ESC cancels phase 2 → back to phase 1
                    self._selected_dep_idx = None
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check standard control buttons first
                for btn in self._buttons:
                    if btn["rect"].collidepoint(event.pos):
                        return btn["action"]
                # Store click for human input processing
                self._last_click_pos = event.pos
        return None

    def handle_human_click(
        self,
        state: GameState,
        departures: List[Departure],
        team_id: str,
    ) -> Optional[dict]:
        """
        Inspect the last stored mouse click and resolve it against the human
        player's available options.

        Phase 1 (no departure selected):
            Returns {"type": "challenge"}  — clicked challenge at own station
                    {"type": "skip"}       — clicked skip button
                    None                   — clicked a destination → select it (phase 2)
            (Destination selection transitions to phase 2; the method returns None.)

        Phase 2 (departure selected):
            Returns {"type": "departure", "idx": int}  — confirmed (chips already set on departure)
                    {"type": "skip"}                   — skip button
                    None                               — +/- clicked or cancel (stays in phase 2/1)
        """
        pos = self._last_click_pos
        self._last_click_pos = None
        if pos is None:
            return None

        # Skip button — available in both phases
        if self._skip_button and self._skip_button["rect"].collidepoint(pos):
            self._selected_dep_idx = None
            return {"type": "skip"}

        # ── Phase 2: chip configuration ──────────────────────────────
        if self._selected_dep_idx is not None:
            dep = departures[self._selected_dep_idx]

            # Confirm
            if self._confirm_btn and self._confirm_btn["rect"].collidepoint(pos):
                n = len(dep.intermediate_stops)
                dep.chips_per_stop = list(self._pending_chips[:n])
                idx = self._selected_dep_idx
                self._selected_dep_idx = None
                return {"type": "departure", "idx": idx}

            # Cancel → back to phase 1
            if self._cancel_btn and self._cancel_btn["rect"].collidepoint(pos):
                self._selected_dep_idx = None
                return None

            # Per-stop +/-
            for btn in self._stop_chip_btns:
                si = btn["stop_idx"]
                if btn["minus_rect"].collidepoint(pos):
                    self._pending_chips[si] = max(0, self._pending_chips[si] - 1)
                    return None
                if btn["plus_rect"].collidepoint(pos):
                    self._pending_chips[si] = min(
                        self._max_chips, self._pending_chips[si] + 1
                    )
                    return None

            # Click elsewhere on panel → cancel (go back to phase 1)
            if not self.map_rect.collidepoint(pos):
                self._selected_dep_idx = None
            return None

        # ── Phase 1: destination / challenge selection ────────────────
        challenge_ids = {c.station_id for c in state.challenges}
        team = state.teams[team_id]
        click_radius = 22

        # Challenge at own station
        own_px = self._station_px.get(team.current_station)
        if own_px:
            if (math.hypot(pos[0] - own_px[0], pos[1] - own_px[1]) <= click_radius
                    and team.current_station in challenge_ids):
                return {"type": "challenge"}

        # Find nearest departure destination
        if self.map_rect.collidepoint(pos):
            best_idx = None
            best_dist = float("inf")
            for i, dep in enumerate(departures):
                dest_px = self._station_px.get(dep.destination_stop_id)
                if dest_px is None:
                    continue
                d = math.hypot(pos[0] - dest_px[0], pos[1] - dest_px[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None and best_dist <= click_radius:
                self._select_departure(best_idx, departures)

        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        state: GameState,
        log: List[str],
        starting_station_id: str,
        human_context=None,
    ) -> None:
        """
        human_context: (team_id, departures) tuple when a human player needs
        to make a decision, or None for AI-only / waiting states.
        """
        self.screen.fill(BG)
        self._draw_edges(state)
        self._draw_stations(state, starting_station_id)
        self._draw_routes(state)
        if human_context is not None:
            team_id, departures = human_context
            self._draw_human_overlay(state, team_id, departures)
        self._draw_panel(state, log, starting_station_id, human_context)
        pygame.display.flip()

    def _draw_edges(self, state: GameState) -> None:
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for a, b in self._edges:
            pa = self._station_px.get(a)
            pb = self._station_px.get(b)
            if pa and pb:
                pygame.draw.line(surf, (*EDGE_COL, 100), pa, pb, 1)
        self.screen.blit(surf, (0, 0))

    def _station_colour(self, station: Station, starting_id: str) -> Tuple[int, int, int]:
        if station.id == starting_id:
            return (180, 180, 60)  # yellow — starting station
        ctrl = station.controlling_team()
        if ctrl == "A":
            return TEAM_A
        elif ctrl == "B":
            return TEAM_B
        elif station.chips_team_a > 0 or station.chips_team_b > 0:
            return CONTESTED
        return NEUTRAL

    def _station_radius(self, station: Station) -> int:
        total = station.chips_team_a + station.chips_team_b
        r = MIN_STATION_R + min(total // 5, MAX_STATION_R - MIN_STATION_R)
        return r

    def _draw_stations(self, state: GameState, starting_id: str) -> None:
        challenge_ids = {c.station_id for c in state.challenges}

        for sid, station in state.stations.items():
            px = self._station_px.get(sid)
            if px is None:
                continue

            colour = self._station_colour(station, starting_id)
            r = self._station_radius(station)

            # Challenge halo
            if sid in challenge_ids:
                pygame.draw.circle(self.screen, CHALLENGE_C, px, r + 5, 2)

            pygame.draw.circle(self.screen, colour, px, r)

            # Chip margin label (e.g. "A+3" or "B+2")
            chips_a = station.chips_team_a
            chips_b = station.chips_team_b
            if chips_a > 0 or chips_b > 0:
                if chips_a > chips_b:
                    margin = chips_a - chips_b
                    label = f"A+{margin}"
                    lc = TEAM_A
                elif chips_b > chips_a:
                    margin = chips_b - chips_a
                    label = f"B+{margin}"
                    lc = TEAM_B
                else:
                    label = f"={chips_a}"
                    lc = CONTESTED
                lsurf = self.font_sm.render(label, True, lc)
                self.screen.blit(lsurf, (px[0] - lsurf.get_width() // 2, px[1] + r + 2))

            # Team position markers (ring around station)
            team_a = state.teams["A"]
            team_b = state.teams["B"]
            if team_a.current_station == sid:
                pygame.draw.circle(self.screen, TEAM_A, px, r + 7, 3)
            if team_b.current_station == sid:
                pygame.draw.circle(self.screen, TEAM_B, px, r + 9, 3)

    def _draw_routes(self, state: GameState) -> None:
        """
        Draw each team's route as a polyline through all remaining intermediate stops.
        """
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        OFFSET = 4
        offsets = {"A": +OFFSET, "B": -OFFSET}

        for team_id, colour in (("A", ROUTE_A), ("B", ROUTE_B)):
            team = state.teams[team_id]
            if not team.is_in_transit():
                continue

            stops = [team.current_station] + list(team.remaining_stops)
            points = [self._station_px.get(s) for s in stops]
            points = [p for p in points if p is not None]
            if len(points) < 2:
                continue

            side = offsets[team_id]

            for i in range(len(points) - 1):
                p0, p1 = points[i], points[i + 1]
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                length = math.hypot(dx, dy) or 1
                ox = -dy / length * side
                oy =  dx / length * side
                s = (int(p0[0] + ox), int(p0[1] + oy))
                d = (int(p1[0] + ox), int(p1[1] + oy))
                pygame.draw.line(surf, colour, s, d, 2)

            # Arrow head at final destination
            if len(points) >= 2:
                p0, p1 = points[-2], points[-1]
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                length = math.hypot(dx, dy) or 1
                ux, uy = dx / length, dy / length
                ox = -dy / length * side
                oy =  dx / length * side
                tip   = (int(p1[0] + ox), int(p1[1] + oy))
                left  = (int(tip[0] - ux * 9 + oy * 0.5), int(tip[1] - uy * 9 - ox * 0.5))
                right = (int(tip[0] - ux * 9 - oy * 0.5), int(tip[1] - uy * 9 + ox * 0.5))
                pygame.draw.polygon(surf, colour, [tip, left, right])

            origin = self._station_px.get(team.current_station)
            if origin:
                pygame.draw.circle(surf, colour, origin, 6)

        self.screen.blit(surf, (0, 0))

    # ------------------------------------------------------------------
    # Human overlay (map layer)
    # ------------------------------------------------------------------

    def _draw_human_overlay(
        self,
        state: GameState,
        team_id: str,
        departures: List[Departure],
    ) -> None:
        team = state.teams[team_id]
        challenge_ids = {c.station_id for c in state.challenges}
        current_wall = sim_minute_to_wall_clock(state.sim_minute)

        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        # Bright ring on human's current station
        own_px = self._station_px.get(team.current_station)
        if own_px:
            pygame.draw.circle(surf, (*HUMAN_PULSE, 200), own_px, 20, 4)

        # Challenge label above own station if present
        if team.current_station in challenge_ids and own_px:
            chal = next(c for c in state.challenges if c.station_id == team.current_station)
            val = chal.current_value()
            ctype = "chip_gain" if chal.type == "chip_gain" else "steal"
            label = f"CHALLENGE +{val:.0f} ({ctype})  click to do"
            lsurf = self.font_md.render(label, True, CHALLENGE_C)
            self.screen.blit(lsurf, (own_px[0] - lsurf.get_width() // 2, own_px[1] - 38))

        self.screen.blit(surf, (0, 0))

        # Destination badges
        badge_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for i, dep in enumerate(departures):
            dest_px = self._station_px.get(dep.destination_stop_id)
            if dest_px is None:
                continue

            is_selected = (self._selected_dep_idx == i)
            minutes_away = dep.arrival_minutes[-1] - current_wall

            # Cost: use current pending_chips if this departure is selected, else zeros
            if is_selected:
                cps = list(self._pending_chips[:len(dep.intermediate_stops)])
            else:
                cps = [0] * len(dep.intermediate_stops)

            cost = compute_route_chip_cost(
                state.stations,
                dep.intermediate_stops,
                team_id,
                self.starting_station_id,
                chips_per_stop=cps,
                max_chip_differential=self._max_chips,
            )
            affordable = team.coins <= 0 or cost <= team.coins

            if is_selected:
                border = OVERLAY_SELECTED
            elif affordable:
                border = OVERLAY_BORDER_OK
            else:
                border = OVERLAY_BORDER_NO

            dest_station = state.stations.get(dep.destination_stop_id)
            name = dest_station.name[:14] if dest_station else dep.destination_stop_id[:14]
            line2 = f"~{minutes_away}m  ¢{cost}"

            # Dim non-selected departures when in phase 2
            alpha = 255 if (self._selected_dep_idx is None or is_selected) else 80

            pad = 4
            w1 = self.font_sm.size(name)[0]
            w2 = self.font_sm.size(line2)[0]
            bw = max(w1, w2) + pad * 2
            bh = self.font_sm.get_height() * 2 + pad * 3
            bx = dest_px[0] - bw // 2
            by = dest_px[1] - bh - 16

            bg_col = (40, 40, 10, int(220 * alpha // 255)) if is_selected else (20, 20, 30, int(200 * alpha // 255))
            pygame.draw.rect(badge_surf, bg_col, (bx, by, bw, bh), border_radius=4)
            bcol_a = (*( CHALLENGE_C if dep.destination_stop_id in challenge_ids else border), alpha)
            pygame.draw.rect(badge_surf, bcol_a, (bx, by, bw, bh), 1 if not is_selected else 2, border_radius=4)

            ta = int(alpha)
            t1 = self.font_sm.render(name, True, OVERLAY_TEXT)
            t2 = self.font_sm.render(line2, True, border)
            t1.set_alpha(ta)
            t2.set_alpha(ta)
            badge_surf.blit(t1, (bx + pad, by + pad))
            badge_surf.blit(t2, (bx + pad, by + pad + self.font_sm.get_height() + 2))

            line_a = int(120 * alpha // 255)
            pygame.draw.line(badge_surf, (*border, line_a),
                             (dest_px[0], dest_px[1] - 16),
                             (dest_px[0], by + bh), 1)

        self.screen.blit(badge_surf, (0, 0))

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _draw_panel(self, state: GameState, log: List[str], starting_id: str, human_context=None) -> None:
        panel_x = self.W - PANEL_W
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, PANEL_W, self.H))

        y = 10
        def text(msg, font=None, colour=TEXT_COL, x=panel_x + 10):
            nonlocal y
            f = font or self.font_md
            surf = f.render(msg, True, colour)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 3

        # Header
        text(f"Day {state.day} / {self.config['game']['num_days']}", self.font_xl, TEXT_COL)
        text(f"Time: {state.sim_minute_to_clock()}", self.font_lg)
        y += 6

        # Team A
        team_a = state.teams["A"]
        ctrl_a = count_controlled_stations(state, "A", starting_id)
        text("── Team A ──", self.font_lg, TEAM_A)
        text(f"  Coins:    {team_a.coins}")
        text(f"  Stations: {ctrl_a}")
        station_a = state.stations.get(team_a.current_station)
        at_name_a = station_a.name[:22] if station_a else "?"
        status_a = "transit" if team_a.is_in_transit() else "idle"
        text(f"  [{status_a}] {at_name_a}")
        if team_a.is_in_transit():
            dst = state.stations.get(team_a.destination_station or "")
            dst_name = dst.name[:24] if dst else "?"
            text(f"  → {dst_name}", colour=TEAM_A)

        y += 4

        # Team B
        team_b = state.teams["B"]
        ctrl_b = count_controlled_stations(state, "B", starting_id)
        text("── Team B ──", self.font_lg, TEAM_B)
        text(f"  Coins:    {team_b.coins}")
        text(f"  Stations: {ctrl_b}")
        station_b = state.stations.get(team_b.current_station)
        at_name_b = station_b.name[:22] if station_b else "?"
        status_b = "transit" if team_b.is_in_transit() else "idle"
        text(f"  [{status_b}] {at_name_b}")
        if team_b.is_in_transit():
            dst = state.stations.get(team_b.destination_station or "")
            dst_name = dst.name[:24] if dst else "?"
            text(f"  → {dst_name}", colour=TEAM_B)

        y += 6

        # Challenges
        text(f"── Challenges ({len(state.challenges)}) ──", self.font_lg, CHALLENGE_C)
        for c in state.challenges[:6]:
            stn = state.stations.get(c.station_id)
            val = c.current_value()
            label = "chip" if c.type == "chip_gain" else "steal"
            name = stn.name[:18] if stn else c.station_id[:18]
            text(f"  [{label}] {name}  +{val:.0f}", self.font_sm)
        if len(state.challenges) > 6:
            text(f"  ... +{len(state.challenges) - 6} more", self.font_sm)

        y += 6

        # Human player status / per-stop controls
        if human_context is not None:
            h_tid, departures = human_context
            h_colour = TEAM_A if h_tid == "A" else TEAM_B

            if self._selected_dep_idx is None:
                # Phase 1 hint
                text(f"── YOUR TURN (Team {h_tid}) ──", self.font_lg, h_colour)
                text("Click a destination to travel", self.font_sm)
                challenge_ids = {c.station_id for c in state.challenges}
                if state.teams[h_tid].current_station in challenge_ids:
                    text("Or click your station for challenge", self.font_sm, CHALLENGE_C)
            else:
                # Phase 2: per-stop chip controls
                dep = departures[self._selected_dep_idx]
                dest_st = state.stations.get(dep.destination_stop_id)
                dest_name = dest_st.name[:20] if dest_st else dep.destination_stop_id[:20]
                text(f"── Trip to {dest_name[:14]} ──", self.font_lg, h_colour)
                text("Coins to drop per stop:", self.font_sm)

                # Draw stop rows (up to MAX_STOP_ROWS)
                n_stops = len(dep.intermediate_stops)
                visible = min(n_stops, MAX_STOP_ROWS)
                for btn in self._stop_chip_btns:
                    si = btn["stop_idx"]
                    stop_id = dep.intermediate_stops[si]
                    stn = state.stations.get(stop_id)
                    sname = (stn.name[:14] if stn else stop_id[:14])
                    coins = self._pending_chips[si] if si < len(self._pending_chips) else 0
                    row_y = btn["row_y"]

                    # Name
                    ns = self.font_sm.render(sname, True, TEXT_COL)
                    self.screen.blit(ns, (panel_x + 10, row_y + 2))

                    # [-] button
                    mc = CHIP_BTN_ACTIVE if coins > 0 else CHIP_BTN_IDLE
                    pygame.draw.rect(self.screen, mc, btn["minus_rect"], border_radius=2)
                    ml = self.font_sm.render("-", True, BTN_TEXT)
                    self.screen.blit(ml, (btn["minus_rect"].x + (btn["minus_rect"].w - ml.get_width()) // 2,
                                         btn["minus_rect"].y + (btn["minus_rect"].h - ml.get_height()) // 2))

                    # Count
                    cl = self.font_sm.render(str(coins), True, OVERLAY_SELECTED if coins > 0 else TEXT_COL)
                    cx = btn["minus_rect"].right + (btn["plus_rect"].left - btn["minus_rect"].right - cl.get_width()) // 2
                    self.screen.blit(cl, (cx, row_y + 2))

                    # [+] button
                    pc = CHIP_BTN_ACTIVE if coins < self._max_chips else CHIP_BTN_IDLE
                    pygame.draw.rect(self.screen, pc, btn["plus_rect"], border_radius=2)
                    pl = self.font_sm.render("+", True, BTN_TEXT)
                    self.screen.blit(pl, (btn["plus_rect"].x + (btn["plus_rect"].w - pl.get_width()) // 2,
                                         btn["plus_rect"].y + (btn["plus_rect"].h - pl.get_height()) // 2))

                if n_stops > MAX_STOP_ROWS:
                    extra = n_stops - MAX_STOP_ROWS
                    last_row_y = self._stop_chip_btns[-1]["row_y"] + STOP_ROW_H + 2
                    ms = self.font_sm.render(f"... +{extra} more stops (0 coins)", True, (130, 130, 140))
                    self.screen.blit(ms, (panel_x + 10, last_row_y))

                # Confirm / Cancel
                if self._confirm_btn:
                    pygame.draw.rect(self.screen, CONFIRM_BTN_COL, self._confirm_btn["rect"], border_radius=4)
                    lbl = self.font_sm.render(self._confirm_btn["label"], True, BTN_TEXT)
                    r = self._confirm_btn["rect"]
                    self.screen.blit(lbl, (r.x + (r.w - lbl.get_width()) // 2, r.y + (r.h - lbl.get_height()) // 2))
                if self._cancel_btn:
                    pygame.draw.rect(self.screen, CANCEL_BTN_COL, self._cancel_btn["rect"], border_radius=4)
                    lbl = self.font_sm.render(self._cancel_btn["label"], True, BTN_TEXT)
                    r = self._cancel_btn["rect"]
                    self.screen.blit(lbl, (r.x + (r.w - lbl.get_width()) // 2, r.y + (r.h - lbl.get_height()) // 2))

        elif self.human_team_id is not None:
            # Human is in transit / doing a challenge
            h_colour = TEAM_A if self.human_team_id == "A" else TEAM_B
            text(f"── Team {self.human_team_id}: in progress... ──", self.font_lg, h_colour)

        # Skip button (always drawn in human mode)
        if self._skip_button and human_context is not None:
            sb = self._skip_button
            pygame.draw.rect(self.screen, SKIP_BTN_COL, sb["rect"], border_radius=4)
            lbl = self.font_sm.render(sb["label"], True, BTN_TEXT)
            lx = sb["rect"].x + (sb["rect"].width - lbl.get_width()) // 2
            ly = sb["rect"].y + (sb["rect"].height - lbl.get_height()) // 2
            self.screen.blit(lbl, (lx, ly))

        # Speed / controls
        speed = self.speed_options[self.speed_idx]
        pause_str = "PAUSED" if state.is_paused else f"{speed}x"
        text(f"── Simulation: {pause_str} ──", self.font_lg)
        text("SPACE=pause  UP/DN=speed", self.font_sm)

        # Buttons
        for btn in self._buttons:
            is_active = (
                (btn["action"] == "pause" and state.is_paused) or
                (btn["action"] == "play"  and not state.is_paused)
            )
            pygame.draw.rect(self.screen, BTN_ACTIVE if is_active else BTN_IDLE, btn["rect"], border_radius=4)
            lbl = self.font_sm.render(btn["label"], True, BTN_TEXT)
            lx = btn["rect"].x + (btn["rect"].width - lbl.get_width()) // 2
            ly = btn["rect"].y + (btn["rect"].height - lbl.get_height()) // 2
            self.screen.blit(lbl, (lx, ly))

        # Event log
        log_y = self.H - LOG_LINES * 15 - 10
        pygame.draw.line(self.screen, (60, 60, 80), (panel_x, log_y - 8), (self.W, log_y - 8), 1)
        for line in log[-(LOG_LINES):]:
            if log_y > self.H - 20:
                break
            surf = self.font_sm.render(line[:38], True, (160, 160, 170))
            self.screen.blit(surf, (panel_x + 6, log_y))
            log_y += 15
