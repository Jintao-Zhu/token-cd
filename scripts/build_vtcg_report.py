"""Assemble the final Markdown report from released artifacts only.

Never invents numbers: every figure is read from an artifact JSON/CSV produced by the
audited pipeline, and missing pieces are reported as missing rather than estimated.
"""
from pathlib import Path
import argparse, csv, json


def load(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text())


def _fmt_ms(value):
    return 'n/a' if value is None else f'{value:.1f} ms'


def _fmt_mib(value):
    return 'n/a' if value is None else f'{value / 1048576:.0f} MiB'


def _engineering_table(art, lock):
    """Preflight success counts per arm/task. Descriptive only, n=2 per cell."""
    tasks, arms, seeds = lock.get('tasks'), lock.get('arms'), lock.get('engineering_seeds')
    if not (tasks and arms and seeds):
        return []
    episodes = len(list((art / 'episodes' / 'engineering').glob('*/*/*_summary.json')))
    if episodes < len(tasks) * len(arms) * len(seeds):
        return []
    rows = []
    for arm in arms:
        cells = []
        for task in tasks:
            ok = n = 0
            for seed in seeds:
                p = art / 'episodes' / 'engineering' / task / arm / f'episode_{seed:03d}_summary.json'
                if p.is_file():
                    n += 1
                    ok += int(json.loads(p.read_text())['success'])
            cells.append((ok, n))
        rows.append((arm, cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    art = a.artifact
    lock = load(art / 'CONFIG_LOCK.json', {})
    eng = load(art / 'ENGINEERING_AUDIT_PASS.json', {})
    mem = load(art / 'FORMAL_MEMBERSHIP_AUDIT.json', {})
    ana = load(art / 'FORMAL_ANALYSIS.json', {})
    par = load(art / 'FORMAL_MODEL_PARITY.json', {})
    L = []
    w = L.append
    w('# OpenVLA 九条件联合实验报告\n')
    w(f'协议 `{lock.get("protocol_id")}`；物理 GPU 白名单 {lock.get("allowed_gpus")}。'
      '所有数字直接来自受审计的 artifact，脚本可复算。\n')
    w('## 1. 冻结协议\n')
    w('| 项 | 值 |\n|---|---|')
    for k, label in [('tasks', '任务'), ('arms', '九条件'), ('formal_seeds', '正式种子'),
                     ('lambda', 'λ（CD 强度）'), ('attention_layers', '注意力层'),
                     ('kmeans_k', 'KMeans K'), ('fixed_m', 'Fixed 预算 m'),
                     ('top_p', 'TopP'), ('top_p_min', 'TopP 下界'),
                     ('top_p_max', 'TopP 上界'), ('prefix', '前缀'), ('intervention', '干预位置')]:
        v = lock.get(k)
        if k == 'formal_seeds' and isinstance(v, list):
            v = f'{v[0]}–{v[-1]}（n={len(v)}）'
        w(f'| {label} | `{v}` |')
    w('')
    w(f"- 预算参数来源：{lock.get('budget_parameter_provenance')}")
    w(f"- 种子解释：{lock.get('seed_interpretation')}")
    w(f"- 种子历史审计：{lock.get('seed_history_audit')}\n")
    w('## 2. 预检与成员审计\n')
    w(f"- 工程预检门禁：`{eng.get('observed_episodes')}/{eng.get('expected_episodes')}` 集，"
      f"通过={not eng.get('reasons')}；说明：{eng.get('limitation', '')}")
    w(f"- 正式成员审计：`{mem.get('observed')}/{mem.get('expected')}` 集，通过={mem.get('passed')}，"
      f"配对 case={mem.get('paired_cases')}，失败项={len(mem.get('failures', []))}")
    if par:
        w(f"- 首步干净分支九条件一致性：完全逐位一致的 case={par.get('exact_cases_with_all_nine_arms')}，"
          f"最大绝对偏差={par.get('worst_max_abs_delta')}（0 表示逐位一致）")
    fs = load(art / 'ENGINEERING_FIRST_STATE_AUDIT.json', {})
    if fs:
        w(f"- 工程同状态预算/排序检查：`{len(fs.get('checked', []))}/{fs.get('expected_checks')}`，"
          f"失败={len(fs.get('failures', []))}")
    eng_rows = _engineering_table(art, lock)
    if eng_rows:
        w('')
        w('**工程预检描述性成功率（每格 n=2，仅验证管线，不作任何结论、不得引用为结果）**\n')
        w('| 条件 | ' + ' | '.join(t.replace('google_robot_', '') for t in lock.get('tasks', []))
          + ' | 合计 |\n' + '|---' * (len(lock.get('tasks', [])) + 2) + '|')
        for arm, cells in eng_rows:
            total_ok = sum(x[0] for x in cells)
            total_n = sum(x[1] for x in cells)
            w(f"| `{arm}` | " + ' | '.join(f'{ok}/{n}' for ok, n in cells)
              + f" | {total_ok}/{total_n} |")
    w('')
    w('## 3. 主要结果\n')
    if not ana:
        w('_正式分析尚未生成：`FORMAL_ANALYSIS.json` 缺失。_\n')
    else:
        w(f"区间：{ana.get('interval')}；每任务 n={ana.get('n_per_task')}。\n")
        w('| 条件 | 成功率 |\n|---|---|')
        for arm, d in ana['arms'].items():
            w(f"| `{arm}` | {d['success']}/{d['n']} = {100*d['success']/d['n']:.1f}% |")
        w('')
        tasks = ana.get('tasks', [])
        if tasks:
            n_per = ana.get('n_per_task')
            w('### 3.1 分任务成功率（每格 n=%s）\n' % n_per)
            w('| 条件 | ' + ' | '.join(t.replace('google_robot_', '') for t in tasks) + ' | 合计 |\n'
              + '|---' * (len(tasks) + 2) + '|')
            for arm, d in ana['arms'].items():
                cells = []
                for t in tasks:
                    v = d['per_task'].get(t, 0)
                    cells.append(f"{v}/{n_per} = {100*v/n_per:.0f}%")
                w(f"| `{arm}` | " + ' | '.join(cells) + f" | {100*d['success']/d['n']:.1f}% |")
            w('')
        names = {'ranking': '排序因子（注意力 vs KMeans）',
                 'budget': '预算因子（matched / fixed / topp）',
                 'structure': '反事实结构对照'}
        for fam, rows in ana['families'].items():
            w(f'### 3.{list(ana["families"]).index(fam)+2} {names.get(fam, fam)}\n')
            w('| 对比（左 − 右） | 差值 | 边际95%区间 | 左胜/右胜 | McNemar p | Holm p |\n|---|---|---|---|---|---|')
            for r in rows:
                w(f"| `{r['left']}` − `{r['right']}` | {r['diff_pp']:+.1f} pp | "
                  f"[{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] | {r['left_only']}/{r['right_only']} | "
                  f"{r['p_exact']:.4g} | {r['p_holm']:.4g} |")
            w('')
        if ana.get('exploratory_interactions'):
            w('### 交互项（探索性，不作确认性推断）\n')
            w('| 对比 | 差值之差 | 95% 区间 |\n|---|---|---|')
            for r in ana['exploratory_interactions']:
                w(f"| {r['budget_comparison']} | {r['difference_of_differences_pp']:+.1f} pp | "
                  f"[{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] |")
            w('')
        if ana.get('vanilla_descriptive'):
            w('### 相对 vanilla 的描述性差异（未做多重比较校正）\n')
            w('| 条件 | 差值 | 95% 区间 |\n|---|---|---|')
            for r in ana['vanilla_descriptive']:
                w(f"| `{r['arm']}` | {r['diff_pp']:+.1f} pp | [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] |")
            w('')
    w('## 4. 失败阶段证据\n')
    st = art / 'FORMAL_FAILURE_STAGES.csv'
    sh = load(art / 'FORMAL_STAGE_SHIFT.json', {})
    if st.is_file():
        rows = list(csv.DictReader(st.open()))
        counts = {}
        for r in rows:
            if r['success'] == 'True':
                continue
            counts.setdefault((r['task'], r['stage']), 0)
            counts[(r['task'], r['stage'])] += 1
        w(f'标签来自模拟器事件流，可由 artifact 复算，**不是人工标注**。失败集 {sum(counts.values())} 条。\n')
        w('| 任务 | 阶段 | 次数 |\n|---|---|---|')
        for (task, stage), n in sorted(counts.items()):
            w(f'| {task} | `{stage}` | {n} |')
        w('')
        if sh:
            w(f"配对阶段位移（vanilla → attn_matched）：{sh.get('note', '')}\n")
    else:
        w('_失败阶段 CSV 尚未生成。_\n')
    w('## 5. 效率\n')
    prof = sorted(p for p in art.glob('profiling*/**/*.json'))
    if prof:
        w('| 文件 | 方法 | 任务/种子 | 同步墙钟均值 | p50 | p95 | 稳态峰值 allocated | 稳态峰值 reserved |\n|---|---|---|---|---|---|---|---|')
        for p in prof:
            d = load(p, {})
            s = d.get('steady') or d.get('aggregate_steady') or {}
            mem = d.get('steady_memory') or {}
            if not mem and d.get('states'):
                mem = d['states'][-1].get('steady_memory', {})
            job = f"{d.get('task') or d.get('factory_arm') or ''}/{d.get('seed', '')}"
            w(f"| `{p.relative_to(art)}` | {d.get('method')} | {job} | "
              f"{_fmt_ms(s.get('mean_ms'))} | {_fmt_ms(s.get('p50_ms'))} | {_fmt_ms(s.get('p95_ms'))} | "
              f"{_fmt_mib(mem.get('peak_allocated_bytes'))} | {_fmt_mib(mem.get('peak_reserved_bytes'))} |")
        w('')
        w('所有计时均来自插桩审计路径，包含完整排序、哈希与审计拷贝，**不能**当作部署速度；'
          '不同方法的解码/前缀差异也使其不能直接当作纯算法开销。测量范围为单一初始帧，'
          '不含仿真步进与渲染；PCD 需要额外状态字段，未实现即不报告数字。\n')
    else:
        w('_效率 profiling 结果尚未生成。_\n')
    w('## 6. 局限（必须与结果一起引用）\n')
    w('- 首帧哈希一致只证明恢复了同一个初始状态与帧，**不证明整条轨迹等价**；'
      '真正的配对证据是首步干净分支动作分数逐位一致（见第 2 节）。')
    w('- Fixed48 与 TopP0.85 来自历史 λ=0.5 扫描中的最佳观测值，属于探索性来源，'
      '**不是独立验证集选择**，不得称为验证集校准。')
    w('- 新种子 500–599 只在有界历史索引中未出现，这不是穷尽性证明；'
      'MoveNear 的初始布局为有限离散集合，不同种子可能采样到相同布局。')
    w('- KMeans 侧是新实现的完整排序比较器，其集合与旧“KMeans winner 并集”不必一致。')
    w('- 区间为边际 95% 配对 bootstrap（任务分层），非联合置信区间；'
      'Holm 校正仅在预先声明的检验族内进行，交互项为探索性。')
    w('- 失败阶段标签由模拟器事件推导，非人工标注；人工标注包已生成但需人类评审。')
    a.out.write_text('\n'.join(L) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
