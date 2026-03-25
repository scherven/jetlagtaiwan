"""
Pygame observer UI for the rail strategy game.

Displays:
  - Rail network map (stations as circles, edges as lines)
  - Chip counts and station control colours
  - Team positions and route lines
  - Challenge markers
  - Score panel (coins, controlled stations, day/time)
  - Pause / Play / Speed controls
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import pygame

from engine.game_state import GameState, Station
from engine.rail_network import RailNetwork
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
OVERLAY_BG       = (20,  20,  30, 200)
OVERLAY_BORDER_OK  = (80, 200,  80)
OVERLAY_BORDER_NO  = (200,  80,  80)
OVERLAY_TEXT     = (240, 240, 240)
CHIP_BTN_ACTIVE  = (100, 200, 100)
CHIP_BTN_IDLE    = (50,  55,  70)
SKIP_BTN_COL     = (160, 120,  40)
HUMAN_PULSE      = (255, 255, 100)   # bright yellow ring on human's station

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
MAP_MARGIN  = 30
PANEL_W     = 280
MIN_STATION_R = 7
MAX_STATION_R = 22
LOG_LINES   = 14


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
        self._max_chips: int = config["game"].get("max_chips_per_station", 5)

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
        self._chip_buttons: List[dict] = []
        self._skip_button: Optional[dict] = None
        self.extra_chips_selected: int = 0
        self._build_buttons()

        # Human click tracking (set during handle_events, consumed by handle_human_click)
        self._last_click_pos: Optional[Tuple[int, int]] = None

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
            # Chip selector: 5 small buttons labelled 0..4
            cbw = (PANEL_W - 20) // 5 - 2
            cbh = 26
            cby = by - 75
            self._chip_buttons = []
            for i in range(5):
                rect = pygame.Rect(bx + i * (cbw + 2), cby, cbw, cbh)
                self._chip_buttons.append({"label": str(i), "value": i, "rect": rect})

            # Skip button
            self._skip_button = {
                "label": "SKIP 30M",
                "rect": pygame.Rect(bx, cby - cbh - 6, PANEL_W - 20, cbh),
                "action": "skip",
            }

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
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check standard control buttons first
                hit_control = False
                for btn in self._buttons:
                    if btn["rect"].collidepoint(event.pos):
                        hit_control = True
                        return btn["action"]
                if not hit_control:
                    # Store click for human input processing
                    self._last_click_pos = event.pos
        return None

    def handle_human_click(
        self,
        state: GameState,
        departures,
        team_id: str,
    ) -> Optional[dict]:
        """
        Inspect the last stored mouse click and resolve it against the human
        player's available options.

        Returns one of:
            {"type": "departure", "idx": int, "extra_chips": int}
            {"type": "challenge"}
            {"type": "skip"}
            {"type": "chips", "value": int}
            None  — click didn't match anything actionable
        """
        pos = self._last_click_pos
        self._last_click_pos = None
        if pos is None:
            return None

        # Check skip button
        if self._skip_button and self._skip_button["rect"].collidepoint(pos):
            return {"type": "skip"}

        # Check chip selector buttons
        for cb in self._chip_buttons:
            if cb["rect"].collidepoint(pos):
                self.extra_chips_selected = cb["value"]
                return {"type": "chips", "value": cb["value"]}

        # Check map area
        if not self.map_rect.collidepoint(pos):
            return None

        # Check if clicking near a departure destination
        click_radius = 22
        challenge_ids = {c.station_id for c in state.challenges}
        team = state.teams[team_id]

        # Check challenge at current station first (click on own station)
        own_px = self._station_px.get(team.current_station)
        if own_px:
            dist = math.hypot(pos[0] - own_px[0], pos[1] - own_px[1])
            if dist <= click_radius and team.current_station in challenge_ids:
                return {"type": "challenge"}

        # Check departure destinations
        best_idx = None
        best_dist = float("inf")
        for i, dep in enumerate(departures):
            dest_px = self._station_px.get(dep.destination_stop_id)
            if dest_px is None:
                continue
            dist = math.hypot(pos[0] - dest_px[0], pos[1] - dest_px[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None and best_dist <= click_radius:
            return {
                "type": "departure",
                "idx": best_idx,
                "extra_chips": self.extra_chips_selected,
            }

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
        to make a decision, or None for AI-only games.
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
            if team_a.current_station == sid:# and not team_a.is_in_transit():
                pygame.draw.circle(self.screen, TEAM_A, px, r + 7, 3)
            if team_b.current_station == sid:# and not team_b.is_in_transit():
                pygame.draw.circle(self.screen, TEAM_B, px, r + 9, 3)

    def _draw_routes(self, state: GameState) -> None:
        """
        Draw each team's route as a polyline through all remaining intermediate stops.
        Segments shared by both teams are offset perpendicularly so both are visible.
        A filled circle marks the team's current position.
        """
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        OFFSET = 4   # pixels perpendicular to each segment
        offsets = {"A": +OFFSET, "B": -OFFSET}

        for team_id, colour in (("A", ROUTE_A), ("B", ROUTE_B)):
            team = state.teams[team_id]
            if not team.is_in_transit():
                continue

            # Build the ordered stop list: current position → remaining stops
            stops = [team.current_station] + list(team.remaining_stops)
            points = [self._station_px.get(s) for s in stops]
            points = [p for p in points if p is not None]
            if len(points) < 2:
                continue

            side = offsets[team_id]

            # Draw each segment with perpendicular offset
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
                tip = (int(p1[0] + ox), int(p1[1] + oy))
                left  = (int(tip[0] - ux * 9 + oy * 0.5), int(tip[1] - uy * 9 - ox * 0.5))
                right = (int(tip[0] - ux * 9 - oy * 0.5), int(tip[1] - uy * 9 + ox * 0.5))
                pygame.draw.polygon(surf, colour, [tip, left, right])

            # Filled circle at current position
            origin = self._station_px.get(team.current_station)
            if origin:
                pygame.draw.circle(surf, colour, origin, 6)

        self.screen.blit(surf, (0, 0))

    # ------------------------------------------------------------------
    # Human overlay
    # ------------------------------------------------------------------

    def _draw_human_overlay(
        self,
        state: GameState,
        team_id: str,
        departures,
    ) -> None:
        """
        Draw departure-destination badges and a highlight ring on the
        human player's current station.
        """
        team = state.teams[team_id]
        team_colour = TEAM_A if team_id == "A" else TEAM_B
        challenge_ids = {c.station_id for c in state.challenges}
        current_wall = sim_minute_to_wall_clock(state.sim_minute)

        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        # Bright pulsing ring on human's current station
        own_px = self._station_px.get(team.current_station)
        if own_px:
            pygame.draw.circle(surf, (*HUMAN_PULSE, 200), own_px, 20, 4)

        # Challenge label above own station if there's one here
        if team.current_station in challenge_ids and own_px:
            chal = next(c for c in state.challenges if c.station_id == team.current_station)
            val = chal.current_value()
            ctype = "chip_gain" if chal.type == "chip_gain" else "steal"
            label = f"CHALLENGE +{val:.0f} ({ctype})"
            lsurf = self.font_md.render(label, True, CHALLENGE_C)
            self.screen.blit(lsurf, (own_px[0] - lsurf.get_width() // 2, own_px[1] - 36))

        self.screen.blit(surf, (0, 0))

        # Departure destination badges
        badge_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        for i, dep in enumerate(departures):
            dest_px = self._station_px.get(dep.destination_stop_id)
            if dest_px is None:
                continue

            minutes_away = dep.arrival_minutes[-1] - current_wall
            cost = compute_route_chip_cost(
                state.stations,
                dep.intermediate_stops,
                team_id,
                self.starting_station_id,
                extra_chips=self.extra_chips_selected,
                max_chips_per_station=self._max_chips,
            )
            affordable = team.coins <= 0 or cost <= team.coins
            border = OVERLAY_BORDER_OK if affordable else OVERLAY_BORDER_NO

            dest_station = state.stations.get(dep.destination_stop_id)
            name = dest_station.name[:14] if dest_station else dep.destination_stop_id[:14]
            line1 = name
            line2 = f"~{minutes_away}m  ¢{cost}"

            # Badge dimensions
            pad = 4
            w1 = self.font_sm.size(line1)[0]
            w2 = self.font_sm.size(line2)[0]
            bw = max(w1, w2) + pad * 2
            bh = self.font_sm.get_height() * 2 + pad * 3
            bx = dest_px[0] - bw // 2
            by = dest_px[1] - bh - 16

            # Background
            pygame.draw.rect(badge_surf, OVERLAY_BG, (bx, by, bw, bh), border_radius=4)
            # Border (brighter if has challenge)
            bcol = (*CHALLENGE_C, 255) if dep.destination_stop_id in challenge_ids else (*border, 200)
            pygame.draw.rect(badge_surf, bcol, (bx, by, bw, bh), 1, border_radius=4)

            # Text
            t1 = self.font_sm.render(line1, True, OVERLAY_TEXT)
            t2 = self.font_sm.render(line2, True, border)
            badge_surf.blit(t1, (bx + pad, by + pad))
            badge_surf.blit(t2, (bx + pad, by + pad + self.font_sm.get_height() + 2))

            # Connector line from badge to station
            pygame.draw.line(badge_surf, (*border, 120),
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

        # Human player extras (chip selector + skip)
        if human_context is not None and self._chip_buttons:
            human_team_id, departures = human_context
            h_colour = TEAM_A if human_team_id == "A" else TEAM_B
            text(f"── YOUR TURN (Team {human_team_id}) ──", self.font_lg, h_colour)
            text("Extra chips/stop:", self.font_sm)
            # Render chip buttons inline
            for cb in self._chip_buttons:
                active = (cb["value"] == self.extra_chips_selected)
                col = CHIP_BTN_ACTIVE if active else CHIP_BTN_IDLE
                pygame.draw.rect(self.screen, col, cb["rect"], border_radius=3)
                lbl = self.font_sm.render(cb["label"], True, BTN_TEXT)
                lx = cb["rect"].x + (cb["rect"].width - lbl.get_width()) // 2
                ly = cb["rect"].y + (cb["rect"].height - lbl.get_height()) // 2
                self.screen.blit(lbl, (lx, ly))
            # Skip button
            if self._skip_button:
                sb = self._skip_button
                pygame.draw.rect(self.screen, SKIP_BTN_COL, sb["rect"], border_radius=4)
                lbl = self.font_sm.render(sb["label"], True, BTN_TEXT)
                lx = sb["rect"].x + (sb["rect"].width - lbl.get_width()) // 2
                ly = sb["rect"].y + (sb["rect"].height - lbl.get_height()) // 2
                self.screen.blit(lbl, (lx, ly))
            y += 8
        elif human_context is None and self.human_team_id is not None:
            # Human is in transit or doing a challenge — just show status
            text("── Waiting... ──", self.font_lg, (150, 150, 180))

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
