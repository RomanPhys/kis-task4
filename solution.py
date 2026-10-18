#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задание №4 «Старая карта». Интерактивная программа экспедиции.

Карта (MAP ... END) приходит на stdin, затем программа печатает команды
LOOK / MOVE <ход> / CLEAR <ход> / END в stdout и читает ответы судьи из stdin.
Алгоритм описан в README.md.
"""
import argparse
import heapq
import math
import random
import sys
import time
from collections import defaultdict, deque


# ----------------------------------------------------------------------------- карта

class GameMap:
    def __init__(self):
        self.day_cost = 0.0
        self.entrance = None
        self.profit = {}                 # зал -> ценность
        self.edges = {}                  # ход -> (зал1, зал2, p)
        self.adj = defaultdict(list)     # зал -> [(ход, сосед)]

    def add_vertex(self, name, profit=0.0):
        self.profit.setdefault(name, profit)

    def other(self, eid, v):
        a, b, _ = self.edges[eid]
        return b if v == a else a


def parse_map(readline):
    """Читает карту до строки END. readline это функция, как у файла."""
    m = GameMap()
    while True:
        raw = readline()
        if raw == '':
            break
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        tok = line.split()
        key = tok[0].lower()
        if key == 'map':
            continue
        if key == 'end':
            break
        if key == 'day_cost':
            m.day_cost = float(tok[1])
        elif key == 'entrance':
            m.entrance = tok[1]
            m.add_vertex(tok[1])
        elif key == 'vertex':
            m.profit[tok[1]] = float(tok[2])
        elif key == 'edge':
            eid, a, b, p = tok[1], tok[2], tok[3], float(tok[4])
            m.add_vertex(a)
            m.add_vertex(b)
            m.edges[eid] = (a, b, min(1.0, max(0.0, p)))
            m.adj[a].append((eid, b))
            if b != a:
                m.adj[b].append((eid, a))
    if m.entrance is None:
        raise ValueError('в карте нет строки entrance')
    return m


# ----------------------------------------------------------------------------- планировщик

class Planner:
    """
    Решает, какой обвалившийся ход расчищать (или закончить экспедицию).

    Стратегии (policy)
      noclear  никогда не расчищать (нижняя оценка)
      index    жадно по «древесному индексу» (базовая эвристика)
      rollout  Монте-Карло rollout поверх базовой эвристики (по умолчанию)
    """

    def __init__(self, m, policy='rollout', budget=0.5, min_samples=32,
                 max_samples=400, top_m=6, max_tree=3000, seed=0):
        self.m = m
        self.policy = policy
        self.budget = budget
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.top_m = top_m
        self.max_tree = max_tree
        self.rng = random.Random(seed)

    # --- базовая эвристика -------------------------------------------------

    def tree_value(self, root, inside):
        """
        Ожидаемая ценность области за пределами `inside`, если оказаться в `root`.

        Строим дерево самых надёжных путей (Дейкстра по весу -ln p) по залам вне inside,
        и на дереве считаем точное значение динамикой снизу вверх:
            F(x)       = profit(x) + sum_{дети y} val(x->y)
            val(x->y)  = p * F(y) + (1-p) * max(0, F(y) - day_cost)
        На дереве с независимыми ходами это точная ценность оптимальной стратегии
        (за обвалом либо платим день, либо бросаем поддерево). На графе это оценка снизу
        по числу маршрутов (альтернативные пути не учитываются).
        """
        m, c = self.m, self.m.day_cost
        dist = {root: 0.0}
        parent = {root: None}
        order, done = [], set()
        pq = [(0.0, root)]
        while pq and len(order) < self.max_tree:
            d, x = heapq.heappop(pq)
            if x in done:
                continue
            done.add(x)
            order.append(x)
            for eid, y in m.adj[x]:
                if y in inside or y in done:
                    continue
                nd = d - math.log(max(m.edges[eid][2], 1e-3))
                if nd < dist.get(y, math.inf):
                    dist[y] = nd
                    parent[y] = eid
                    heapq.heappush(pq, (nd, y))
        F = {x: m.profit[x] for x in order}
        for x in reversed(order):           # дети обрабатываются раньше родителей
            eid = parent[x]
            if eid is None:
                continue
            up = m.other(eid, x)
            p = m.edges[eid][2]
            F[up] += p * F[x] + (1.0 - p) * max(0.0, F[x] - c)
        return F[root]

    def frontier(self, inside, state):
        """Известно обвалившиеся ходы из достижимой области наружу."""
        res = []
        for v in inside:
            for eid, w in self.m.adj[v]:
                if w not in inside and state.get(eid) is False:
                    res.append((eid, w))
        return res

    def index(self, inside, state):
        c = self.m.day_cost
        cache, out = {}, {}
        for eid, w in self.frontier(inside, state):
            if w not in cache:
                cache[w] = self.tree_value(w, inside)
            out[eid] = cache[w] - c
        return out

    # --- симуляция для rollout ---------------------------------------------

    def _expand(self, inside, st, world, start):
        """Бесплатный обход: всё, что достижимо из start по целым ходам."""
        m = self.m
        if start in inside:
            return 0.0
        gain = m.profit[start]
        inside.add(start)
        dq = deque([start])
        while dq:
            x = dq.popleft()
            for eid, y in m.adj[x]:
                if eid not in st:
                    st[eid] = world[eid]
                if st[eid] and y not in inside:
                    inside.add(y)
                    gain += m.profit[y]
                    dq.append(y)
        return gain

    def simulate(self, inside, state, world, first_edge):
        """Выигрыш (без уже собранного) от расчистки first_edge и дальнейшей игры базовой эвристикой."""
        m, c = self.m, self.m.day_cost
        inside, st = set(inside), dict(state)
        gain = 0.0
        e = first_edge
        while e is not None:
            gain -= c
            st[e] = True
            a, b, _ = m.edges[e]
            start = b if a in inside else a
            gain += self._expand(inside, st, world, start)
            idx = self.index(inside, st)
            e = None
            if idx:
                best = max(idx, key=idx.get)
                if idx[best] > 0:
                    e = best
        return gain

    # --- решение -------------------------------------------------------------

    def choose(self, inside, state):
        """Возвращает id хода для CLEAR или None (END)."""
        if self.policy == 'noclear':
            return None
        idx = self.index(inside, state)
        if not idx:
            return None
        # Отсечение: вся оставшаяся ценность снаружи не окупает даже один день.
        outside_total = sum(p for v, p in self.m.profit.items() if v not in inside)
        if outside_total <= self.m.day_cost:
            return None
        ranked = sorted(idx, key=idx.get, reverse=True)
        if self.policy == 'index':
            return ranked[0] if idx[ranked[0]] > 0 else None

        cands = ranked[:self.top_m]
        unknown = [eid for eid in self.m.edges if eid not in state]
        totals = dict.fromkeys(cands, 0.0)
        n, t0 = 0, time.time()
        while n < self.max_samples:
            # общий сэмпл мира для всех кандидатов (common random numbers, так меньше шум сравнения)
            world = {eid: self.rng.random() < self.m.edges[eid][2] for eid in unknown}
            for e in cands:
                totals[e] += self.simulate(inside, state, world, e)
            n += 1
            if not unknown:           # мир уже полностью известен, симуляция детерминирована
                break
            if n >= self.min_samples and time.time() - t0 > self.budget:
                break
        q = {e: totals[e] / n for e in cands}
        best = max(q, key=q.get)
        return best if q[best] > 0 else None


# ----------------------------------------------------------------------------- экспедиция

class Expedition:
    def __init__(self, m, planner, inp, out, log=None):
        self.m = m
        self.planner = planner
        self.inp = inp
        self.out = out
        self.log = log
        self.pos = m.entrance
        self.looked = set()
        self.state = {}          # ход -> True (цел/расчищен) / False (обвал); нет ключа, значит неизвестно
        self.days = 0

    def _log(self, msg):
        if self.log:
            print(msg, file=self.log, flush=True)

    def send(self, cmd):
        self.out.write(cmd + '\n')
        self.out.flush()

    def recv(self):
        while True:
            line = self.inp.readline()
            if line == '':
                raise EOFError('судья закрыл поток')
            line = line.strip()
            if line:
                return line

    # --- команды -------------------------------------------------------------

    def look(self):
        self.send('LOOK')
        incident = {eid for eid, _ in self.m.adj[self.pos]}
        for _ in range(len(incident)):
            tok = self.recv().split()
            if len(tok) >= 3 and tok[0].upper() == 'PATH':
                self.state[tok[1]] = tok[-1].upper() == 'OK'
        self.looked.add(self.pos)

    def move(self, eid):
        self.send(f'MOVE {eid}')
        if self.recv().upper() == 'OK':
            self.state[eid] = True
            self.pos = self.m.other(eid, self.pos)
            return True
        self.state[eid] = False
        return False

    def clear(self, eid):
        self.send(f'CLEAR {eid}')
        self.recv()
        self.state[eid] = True
        self.days += 1
        self.pos = self.m.entrance

    # --- логика --------------------------------------------------------------

    def inside(self):
        """Залы, достижимые от входа по известно целым ходам."""
        seen = {self.m.entrance}
        dq = deque([self.m.entrance])
        while dq:
            x = dq.popleft()
            for eid, y in self.m.adj[x]:
                if self.state.get(eid) and y not in seen:
                    seen.add(y)
                    dq.append(y)
        return seen

    def path_to(self, targets):
        """BFS по целым ходам от текущего зала до ближайшего зала из targets."""
        prev = {self.pos: None}
        dq = deque([self.pos])
        while dq:
            x = dq.popleft()
            if x in targets:
                path = []
                while prev[x] is not None:
                    eid, px = prev[x]
                    path.append(eid)
                    x = px
                return path[::-1]
            for eid, y in self.m.adj[x]:
                if self.state.get(eid) and y not in prev:
                    prev[y] = (eid, x)
                    dq.append(y)
        return None

    def explore(self):
        """Бесплатная фаза: обойти и осмотреть всю достижимую область."""
        while True:
            if self.pos not in self.looked:
                self.look()
                continue
            inside = self.inside()
            targets = inside - self.looked
            if not targets:
                return inside
            path = self.path_to(targets)
            if not path:
                return inside
            for eid in path:
                if not self.move(eid):
                    break

    def run(self):
        while True:
            inside = self.explore()
            eid = self.planner.choose(inside, self.state)
            if eid is None:
                break
            self._log(f'# day {self.days + 1}: CLEAR {eid}')
            self.clear(eid)
        self.send('END')
        try:
            result = self.recv()
        except EOFError:
            result = ''
        self._log(f'# {result}')
        return result


def main():
    ap = argparse.ArgumentParser(description='Задание 4 «Старая карта»')
    ap.add_argument('--policy', default='rollout', choices=['rollout', 'index', 'noclear'])
    ap.add_argument('--budget', type=float, default=0.5, help='секунд на одно решение о расчистке')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--map', help='читать карту из файла, а не из stdin')
    ap.add_argument('--verbose', action='store_true', help='лог решений в stderr')
    args = ap.parse_args()

    for s in (sys.stdin, sys.stdout):
        try:
            s.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    if args.map:
        with open(args.map, encoding='utf-8') as f:
            m = parse_map(f.readline)
    else:
        m = parse_map(sys.stdin.readline)

    planner = Planner(m, policy=args.policy, budget=args.budget, seed=args.seed)
    Expedition(m, planner, sys.stdin, sys.stdout,
               log=sys.stderr if args.verbose else None).run()


if __name__ == '__main__':
    main()
