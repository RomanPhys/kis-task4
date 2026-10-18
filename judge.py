#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Локальный судья для задания 4. Генерирует (или берёт) мир, запускает решение как
подпроцесс и общается с ним по протоколу из условия.

  python judge.py examples/example.map                  # случайный мир
  python judge.py examples/example.map --world x1=1,x2=0,x3=1
  python judge.py --random 60 --games 20 --policy index  # сравнение стратегий
"""
import argparse
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solution import parse_map  # noqa: E402


def random_map(n, seed):
    rng = random.Random(seed)
    lines = ['MAP', f'day_cost {rng.choice([2, 4, 8, 15])}', 'entrance V0', 'vertex V0 0']
    for i in range(1, n):
        lines.append(f'vertex V{i} {rng.choice([0, 0, 3, 5, 10, 20, 40])}')
    eid = 0
    for i in range(1, n):                               # остовное дерево
        lines.append(f'edge e{eid} V{rng.randrange(i)} V{i} {rng.uniform(0.2, 0.95):.2f}')
        eid += 1
    for _ in range(n // 3):                             # плюс несколько циклов
        a, b = rng.sample(range(n), 2)
        lines.append(f'edge e{eid} V{a} V{b} {rng.uniform(0.2, 0.95):.2f}')
        eid += 1
    lines.append('END')
    return '\n'.join(lines) + '\n'


def play(map_text, world, policy, budget, verbose=False):
    m = parse_map(_reader(map_text))
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, 'solution.py'),
                             '--policy', policy, '--budget', str(budget)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding='utf-8')
    proc.stdin.write(map_text)
    proc.stdin.flush()
    pos, reached, days = m.entrance, {m.entrance}, 0
    state = dict(world)

    def say(s):
        if verbose:
            print('  ' + s)
        proc.stdin.write(s + '\n')
        proc.stdin.flush()

    while True:
        cmd = proc.stdout.readline().strip()
        if not cmd:
            raise RuntimeError('решение замолчало')
        if verbose:
            print('> ' + cmd)
        tok = cmd.split()
        if tok[0] == 'LOOK':
            for e, w in m.adj[pos]:
                say(f'PATH {e} {w} {"OK" if state[e] else "BROKEN"}')
        elif tok[0] == 'MOVE':
            e = tok[1]
            if e in m.edges and pos in m.edges[e][:2] and state[e]:
                pos = m.other(e, pos)
                reached.add(pos)
                say('OK')
            else:
                say('FAIL')
        elif tok[0] == 'CLEAR':
            state[tok[1]] = True
            days += 1
            pos = m.entrance
            say('OK')
        elif tok[0] == 'END':
            got = sum(m.profit[v] for v in reached)
            res = got - days * m.day_cost
            say(f'RESULT {got:g} {days} {res:g}')
            proc.wait(timeout=5)
            return got, days, res


def _reader(text):
    lines = iter(text.splitlines(True))
    return lambda: next(lines, '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map', nargs='?')
    ap.add_argument('--world', help='например x1=1,x2=0')
    ap.add_argument('--random', type=int, help='сгенерировать карту с N залами')
    ap.add_argument('--games', type=int, default=1)
    ap.add_argument('--policy', default='rollout')
    ap.add_argument('--budget', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    rng = random.Random(args.seed)
    total = 0.0
    for g in range(args.games):
        if args.random:
            text = random_map(args.random, args.seed + g)
        else:
            with open(args.map, encoding='utf-8') as f:
                text = f.read()
        m = parse_map(_reader(text))
        if args.world:
            world = {kv.split('=')[0]: kv.split('=')[1] == '1' for kv in args.world.split(',')}
        else:
            world = {e: rng.random() < p for e, (_, _, p) in m.edges.items()}
        got, days, res = play(text, world, args.policy, args.budget, verbose=args.games == 1)
        total += res
        if args.games > 1:
            print(f'game {g}: collected {got:g}, days {days}, profit {res:g}')
    if args.games > 1:
        print(f'mean profit ({args.policy}): {total / args.games:.2f}')


if __name__ == '__main__':
    main()
