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
from engine.rules import count_controlled_stations

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

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
MAP_MARGIN  = 40
PANEL_W     = 260
MIN_STATION_R = 4
MAX_STATION_R = 14
LOG_LINES   = 12


class Display:
    """
    Manages the Pygame window and renders the game state.

    Call display.handle_events() + display.draw(state, net, log) each frame.
    """

    def __init__(self, config: dict, rail_network: RailNetwork, starting_station_id: str):
        self.config = config
        self.net = rail_network
        self.starting_station_id = starting_station_id
        self.speed_options: List[float] = config["simulation"]["speed_options"]

        pygame.init()
        info = pygame.display.Info()
        self.W = min(1400, info.current_w - 40)
        self.H = min(820,  info.current_h - 60)
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
        self._build_buttons()

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
            {"label": "⏸ Pause",  "rect": pygame.Rect(bx, by, bw, bh),       "action": "pause"},
            {"label": "▶ Play",   "rect": pygame.Rect(bx + bw + 10, by, bw, bh), "action": "play"},
            {"label": "Speed ▲",  "rect": pygame.Rect(bx, by + bh + 8, bw, bh),  "action": "speed_up"},
            {"label": "Speed ▼",  "rect": pygame.Rect(bx + bw + 10, by + bh + 8, bw, bh), "action": "speed_down"},
        ]

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
        """
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
                for btn in self._buttons:
                    if btn["rect"].collidepoint(event.pos):
                        return btn["action"]
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(
        self,
        state: GameState,
        log: List[str],
        starting_station_id: str,
    ) -> None:
        self.screen.fill(BG)
        self._draw_edges(state)
        self._draw_stations(state, starting_station_id)
        self._draw_routes(state)
        self._draw_panel(state, log, starting_station_id)
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

            # Team position markers (ring around station)
            team_a = state.teams["A"]
            team_b = state.teams["B"]
            if team_a.current_station == sid and not team_a.is_in_transit():
                pygame.draw.circle(self.screen, TEAM_A, px, r + 7, 3)
            if team_b.current_station == sid and not team_b.is_in_transit():
                pygame.draw.circle(self.screen, TEAM_B, px, r + 9, 3)

    def _draw_routes(self, state: GameState) -> None:
        """Draw a line from each team's current position to their destination."""
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for team_id, colour in (("A", ROUTE_A), ("B", ROUTE_B)):
            team = state.teams[team_id]
            if not team.is_in_transit():
                continue
            src = self._station_px.get(team.current_station)
            dst = self._station_px.get(team.destination_station) if team.destination_station else None
            if src and dst:
                pygame.draw.line(surf, colour, src, dst, 3)
                # Arrow head at destination
                dx, dy = dst[0] - src[0], dst[1] - src[1]
                length = math.hypot(dx, dy) or 1
                ux, uy = dx / length, dy / length
                tip = dst
                left  = (int(tip[0] - ux * 10 + uy * 5), int(tip[1] - uy * 10 - ux * 5))
                right = (int(tip[0] - ux * 10 - uy * 5), int(tip[1] - uy * 10 + ux * 5))
                pygame.draw.polygon(surf, colour, [tip, left, right])
        self.screen.blit(surf, (0, 0))

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _draw_panel(self, state: GameState, log: List[str], starting_id: str) -> None:
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
        text(f"  At:       {station_a.name[:22] if station_a else '?'!r}")
        if team_a.is_in_transit():
            dst = state.stations.get(team_a.destination_station or "")
            text(f"  → {dst.name[:20] if dst else '?'!r}")

        y += 4

        # Team B
        team_b = state.teams["B"]
        ctrl_b = count_controlled_stations(state, "B", starting_id)
        text("── Team B ──", self.font_lg, TEAM_B)
        text(f"  Coins:    {team_b.coins}")
        text(f"  Stations: {ctrl_b}")
        station_b = state.stations.get(team_b.current_station)
        text(f"  At:       {station_b.name[:22] if station_b else '?'!r}")
        if team_b.is_in_transit():
            dst = state.stations.get(team_b.destination_station or "")
            text(f"  → {dst.name[:20] if dst else '?'!r}")

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

        # Speed / controls
        speed = self.speed_options[self.speed_idx]
        pause_str = "PAUSED" if state.is_paused else f"{speed}x"
        text(f"── Simulation: {pause_str} ──", self.font_lg)
        text("SPACE=pause  ↑↓=speed", self.font_sm)

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
