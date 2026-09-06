"""Export the three-arm comparison report from the executed replication notebook, without transcribed tables.

Called by build_three_arm_notebook.py's generated notebook. The namespace must
contain all completed analyses. This module never changes experiment artefacts.
Edit explanatory prose here; the Markdown report is generated.
"""
from __future__ import annotations

import hashlib
import json
import platform
import statistics as st
from collections import Counter
from importlib.metadata import version
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr


def _records(mapping, names):
    """Turn tuple-keyed analysis tables into JSON rows with explicit dimensions."""
    return [dict(zip(names, key), **value) for key, value in mapping.items()]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def export_report(ns):
    repo, data_dir = ns['REPO'], ns['DATA']
    bench, arms, policies = ns['BENCH'], ns['ORDER'], ns['policies']
    baseline, rp, per_rep = ns['baseline'], ns['RP'], ns['per_rep']
    between, holm = ns['between'], ns['holm_adjust']
    keys = [('F1', 'mean_cl', 'lo'), ('F2', 'mean_correctness', 'hi'),
            ('F3', 'mean_energy', 'lo'), ('feasible', 'n_feasible', 'hi')]
    short = {b: 'C' if 'Codeflaws' in b else 'Python' for b in bench}
    out = []

    def para(text):
        out.append(text.strip())

    def table(headers, rows):
        out.append('\n'.join(['| ' + ' | '.join(headers) + ' |',
                              '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                             ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]))

    def f(value, digits=4):
        return f'{value:.{digits}f}'

    def pm(values, digits=4):
        return f'{st.mean(values):.{digits}f} ± {st.stdev(values):.{digits}f}'

    def fig(name, caption):
        para(f'![{caption}](../../analysis/figs/fig_{name}.png)\n\n{caption}')

    aggregates, joint, efficiency = [], [], []
    for b in bench:
        for a in arms:
            recs = [r for i in rp(b, a) for r in policies[b][a][i]]
            for label, key, _ in keys:
                vals = ns['flat'](b, a, key)
                reps = per_rep(b, a, key)
                aggregates.append(dict(benchmark=b, arm=a, outcome=label,
                    per_rep_mean=reps, per_rep_median=per_rep(b, a, key, st.median),
                    mean=st.mean(reps), sd=st.stdev(reps),
                    cv_pct=100*st.stdev(reps)/abs(st.mean(reps)),
                    pooled_mean=st.mean(vals), pooled_median=st.median(vals),
                    pooled_skew=float(np.mean((np.array(vals)-np.mean(vals))**3)/(np.std(vals)**3+1e-12))))
            bs = baseline[b]['summary']
            n_joint = sum(r['mean_cl'] < bs['mean_cl']['mean'] and
                          r['mean_correctness'] > bs['mean_correctness']['mean'] and
                          r['mean_energy'] < bs['mean_energy']['mean'] for r in recs)
            joint.append(dict(benchmark=b, arm=a, count=n_joint, total=len(recs),
                              percent=100*n_joint/len(recs)))
            # Retain the original mean of per-policy ratios (not a ratio of pooled means).
            efficiency.append(dict(benchmark=b, arm=a,
                ratio=st.mean(r['mean_correctness']/(r['mean_energy']/1000) for r in recs),
                ratio_of_means=st.mean(r['mean_correctness'] for r in recs) /
                      (st.mean(r['mean_energy'] for r in recs)/1000)))
        br = baseline[b]['reps']
        efficiency.append(dict(benchmark=b, arm='Baseline',
            ratio=st.mean(r['mean_correctness']/(r['mean_energy']/1000) for r in br),
            ratio_of_means=st.mean(r['mean_correctness'] for r in br) /
                  (st.mean(r['mean_energy'] for r in br)/1000)))

    whole = {}
    for name, agg in [('mean', st.mean), ('median', st.median)]:
        whole[name] = [dict(zip(['benchmark', 'arm', 'outcome', 'baseline', 'front',
            'margin', 'relative_pct', 'baseline_sd_units', 'p', 'p_holm'], row))
            for row in ns['whole_front_test'](agg)]
    selection = _records(ns['SELECT_TESTS'], ['rule', 'benchmark', 'arm', 'outcome'])
    selected = [r for r in selection if r['rule'] == 'correctness only']
    cross_arm = _records({k:v for k,v in ns['repl'].items() if len(k)==4},
                         ['benchmark', 'objective_key', 'arm_x', 'arm_y'])

    # Post-hoc sensitivity excludes C records on tasks also present in training.
    # Selection remains fixed by ORIGINAL training fitness; no held-out reranking.
    sensitivity = []
    b_c = next(b for b in bench if 'Codeflaws' in b)
    excluded = set(ns['SPLIT_AUDIT'][b_c]['overlapping_record_ids'])

    def subset_metrics(record):
        pp = [q for q in ns['problem_records'](record) if q['problem_id'] not in excluded]
        feasible = [q for q in pp if q.get('feasible')]
        return [st.mean(q['cl'] for q in feasible), st.mean(q['correctness'] for q in pp),
                st.mean(q['runtime_mj'] for q in feasible), 100*len(feasible)/len(pp)]

    base_sub = np.array([subset_metrics(r) for r in baseline[b_c]['reps']])
    for a in arms:
        rows = []
        for i in rp(b_c, a):
            records, _, training = ns['_mats'](b_c, a, i)
            chosen = ns['rank_by'](training, 'correctness only')[:ns['KSEL']]
            rows.append(np.mean([subset_metrics(records[j]) for j in chosen], axis=0))
        values = np.array(rows)
        ps = [between(values[:,k], base_sub[:,k])[0] for k in range(4)]
        for k, ((label, _, better), p, pa) in enumerate(zip(keys, ps, holm(ps))):
            bm, sm = base_sub[:,k].mean(), values[:,k].mean()
            sensitivity.append(dict(arm=a, outcome=label, records=100-len(excluded),
                baseline=float(bm), selected=float(sm),
                margin=float(bm-sm if better=='lo' else sm-bm), p=p, p_holm=pa))

    inputs = set()
    for b, cfg in bench.items():
        bd = cfg['base_dir']
        inputs.update([bd/'manifest.json', bd/'summary.json'])
        inputs.update((bd/'reps').glob('*.json'))
        for a in arms:
            for i in rp(b,a):
                p = ns['rep_path'](b,a,i)
                inputs.update([p,p.with_name('manifest.json'),repo/cfg[a.lower()+'_front'].format(i=i)])
    for a, (template, _) in ns['TRAJ_SRC'].items():
        inputs.update(repo/template.format(i=i) for i in ns['traj_reps'](a))
    sources = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(inputs)}
    exported = dict(schema_version=1, generated_by='scripts/generators/export_report.py',
        source_sha256=sources, software_versions=dict(python=platform.python_version(),
            **{name:version(name) for name in ['numpy','scipy','matplotlib','pymoo']}), split_audit=ns['SPLIT_AUDIT'], baseline=baseline_summary(baseline),
        front_cardinality={b:{a:[len(policies[b][a][i]) for i in rp(b,a)] for a in arms} for b in bench},
        aggregates=aggregates, whole_front_tests=whole, cross_arm_tests=cross_arm,
        marginal_shares=_records({k:dict(percent=v) for k,v in ns['share_tbl'].items()}, ['benchmark','outcome','arm']),
        simultaneous_improvement=joint, selection_tests=selection,
        c_unseen_task_sensitivity=sensitivity,
        hypervolume=dict(method='Monte Carlo', samples=ns['HV_SAMPLES'], seed=ns['HV_SEED'],
            reference_margin=ns['REF_MARGIN'], subsample_size=ns['K'], subsample_draws=ns['DRAWS'],
            raw=_records({k:dict(per_rep=v) for k,v in ns['hv_vals'].items()}, ['benchmark','arm']),
            cardinality_controlled=_records({k:dict(per_rep=v) for k,v in ns['SIZEMATCH'].items()}, ['benchmark','arm'])),
        indicators=_records(ns['IND'],['benchmark','arm']),
        task_contrasts=_records(ns['contrasts'],['benchmark','arm_x','arm_y','objective']),
        train_heldout_correlations=_records(ns['TRAIN_RANK'],['benchmark','arm','objective']),
        training_shortlist_agreement=_records(ns['TRAIN_AGREEMENT'],['rule_x','rule_y']),
        oracle_agreement=_records({k:dict(identical=v,cells=ns['N_CELLS']) for k,v in ns['ORACLE_AGREEMENT'].items()},['rule_x','rule_y']),
        harmonic_selection=dict(mean_overlap=st.mean(ns['ov']), chance_overlap=st.mean(ns['exp']),
            zero_overlap=sum(v==0 for v in ns['ov']), cells=len(ns['ov']), mean_rho=st.mean(ns['rhos']),
            negative_rho=sum(v<0 for v in ns['rhos'])),
        random_shortlist_reference=_records(ns['RANDOM_SELECTION'],['benchmark','arm']),
        anytime=[dict(arm=a, rep=i, checkpoints=ns['CHECK'], hv=ns['CURVES'][(a,i)],
            thresholds=ns['reach'][(a,i)], auc=ns['auc'][(a,i)], plateau=ns['plat'][(a,i)])
            for a in arms for i in ns['traj_reps'](a)],
        anytime_tests=_records(ns['ANYTIME_TESTS'],['arm_x','arm_y','outcome']),
        cross_language=ns['CROSS_LANGUAGE'], task_structure=ns['PROBLEM_SUMMARY'], efficiency=efficiency,
        equivalence=[dict(benchmark=k[0],objective=k[1],arm_x=k[2],arm_y=k[3],
            difference=v[0],margin=v[1],p_tost=v[2],p_mwu=v[3],verdict=v[4]) for k,v in ns['TOSTR'].items()],
        omnibus=[dict(benchmark=b,objective=k,H=v[0],p=v[1],mean_ranks=v[2]) for (b,k),v in ns['KW'].items()],
        pca_two_component_variance=float(ns['evr'].sum()))
    data_path = data_dir/'three_arm_results.json'

    para('# Replicated Held-Out Comparison of NSGA-II, Random Search and MOBO')
    para('''> Generated from the executed `three_arm_analysis.ipynb` by
`scripts/generators/export_report.py`. Edit the generator/exporter and rerun the
notebook to update this report. Numeric tables are calculated from all five
repetitions, including MOBO reps 0–4; none is manually transcribed.
The companion [full-precision results](../../analysis/data/three_arm_results.json)
includes per-repetition values, test families and hashes of every data input.''')
    para('## 1. Scope and status')
    para('''This report compares three search arms at a matched budget on Codeflaws (C)
and transfers the same retained policies to RunBugRun (Python). All five
repetitions per arm are complete. The default policy has five evaluation
repetitions per benchmark. This revision supersedes earlier three-MOBO-repetition
numbers and the partly updated prose and tables in the previous revision of this report.

The principal result is an exploratory **training-correctness-selected top-five
shortlist** advantage in correctness and feasibility. It does not establish a
universal advantage of optimisation, a single deployable policy, or savings in
total pipeline energy. Detailed results and multiplicity qualifications follow.''')
    para('## 2. Configuration as executed')
    table(['Item','Value'],[
        ['Model','qwen2.5-coder:1.5b'],['Search space','11 decoding/retry parameters across Analyzer, Repairer, Verifier and Optimizer'],
        ['GA','NSGA-II, population 50, 15 generations; seeds 42–46'],
        ['RS','750 uniform random candidates; seeds 99–103'],
        ['MOBO','qLogNEHVI; 50 initial random candidates, 700 subsequent candidates; seeds 123–127'],
        ['Training','The same 50 C bug records for all search repetitions'],
        ['Held-out','100 records per benchmark; see the task-overlap audit below'],
        ['Evaluation','v3.1 two-phase measurement; real rapl_sysfs; 4 problem workers'],
        ['F1 ↓','Mean whole-program cyclomatic complexity over feasible records'],
        ['F2 ↑','Mean test-pass fraction over all records, including zero for infeasible records'],
        ['F3 ↓','Mean repaired-program execution energy, mJ, over feasible records'],
        ['Feasibility ↑','Records with correctness > 0, out of 100; not full repair success']])
    para('F3 excludes LLM search/repair energy and is conditional on feasibility. Lower F3 with a different feasible subset is not automatically an energy saving for the same work. Means across repetitions use equal repetition weights; pooled-policy summaries weight larger fronts more heavily.')
    table(['Arm','rep 0','rep 1','rep 2','rep 3','rep 4','Total per benchmark'],
          [[a,*exported['front_cardinality'][b_c][a],sum(exported['front_cardinality'][b_c][a])] for a in arms])
    para('## 3. Provenance and integrity')
    para('The notebook asserts five complete repetitions per arm and five baseline repetitions. It checks manifest digests, backend, real measurement and worker count; matches every held-out policy vector and training fitness to the saved search front; verifies all 100 record IDs per evaluation; and recomputes F1, F2, F3 and feasibility from per-record payloads. All checks passed.')
    table(['Benchmark','Held-out records','Underlying tasks','Tasks also in excluded training split','Affected held-out records'],
          [[short[b],v['records'],v['groups'],len(v['overlapping_groups']),len(v['overlapping_record_ids'])] for b,v in ns['SPLIT_AUDIT'].items()])
    para('''**C is record-disjoint but not task-disjoint.** Its 50 training records cover
49 Codeforces tasks. Six task prefixes occur in both training and held-out data:
`14-B`, `149-B`, `298-B`, `304-A`, `43-B`, `680-B`; they account for seven held-out
records. Stored C `group_id` values were bug-record IDs, so trusting them had
incorrectly implied full task separation. The corrected task bootstrap uses the
prefix before `-bug-`. This changes task-level analyses, not the original
100-record per-policy outcomes or training selections.

Python's 100 records cover 68 upstream tasks, with no overlap with the excluded
50-record/44-task Python reference split. Candidate records were sampled after
group exclusion; tasks were not sampled uniformly. The optimisation training
itself used C. A fresh C split excluding all training tasks is needed for a fully
task-disjoint confirmatory experiment; the post-hoc sensitivity in §10 is not a
replacement for that experiment.''')
    table(['Benchmark','Held-out SHA-256'],[[short[b],f"`{cfg['digest']}`"] for b,cfg in bench.items()])
    para('## 4. The unit of replication')
    para('''Search-strategy inference uses five independent **search repetitions**, not
hundreds of policies from the same trajectories. Primary tests are two-sided
Mann–Whitney U on per-repetition summaries; Welch tests are sensitivity checks.
SciPy uses exact MWU for these small untied samples and an asymptotic tie
correction when tied. The untied minimum is 2/C(10,5) = 0.00794; Holm over four
outcomes can reach 0.03175, and Holm over three can reach 0.02381.

Pooled-policy comparisons are descriptive diagnostics of the evaluated policy
distributions. Their p-values do not establish independent method-level evidence.
The ratios of repetition-level to pooled-policy p-values depend on both the test
and contrast; they are not a universal effective-sample-size correction.''')
    r=ns['repl'][(b_c,'mean_correctness','GA','RS')]
    para(f"For C F2, GA versus RS has pooled-policy MWU p={r['p_pol']:.6g} and repetition-level MWU p={r['p_rep']:.6g}, a ratio of approximately {r['p_rep']/r['p_pol']:,.0f}, not the previously stated 160,000. Cliff's δ accompanies the same repetition-level unit as the primary test.")
    fig('replication_unit_pvalues','Policy-level versus repetition-level p-values; pooled policies are dependent.')
    para('## 5. Held-out aggregates and estimator sensitivity')
    para('The following are means ± sample SD of the five per-repetition front means. Baseline rows summarise its five repeats. These marginal centres need not correspond to a real policy: each mean coordinate averages all policies, rather than selecting a coordinate from one policy.')
    rows=[]
    for b in bench:
        for a in ['Baseline',*arms]:
            rows.append([short[b],a,*[pm([r[k] for r in baseline[b]['reps']] if a=='Baseline' else per_rep(b,a,k), 2 if k=='n_feasible' else 4) for _,k,_ in keys]])
    table(['Benchmark','Arm','F1 ↓','F2 ↑','F3 mJ ↓','Feasible /100 ↑'],rows)
    para('For comparison, the next table pools all retained policies and gives '
         '**mean / median**, with larger fronts receiving more weight. These are '
         'different estimands from the equally weighted repetition means above. '
         'In particular, the updated MOBO C pooled energy mean must not be confused '
         'with its mean of five repetition-level means.')
    table(['Benchmark','Arm','Pooled F1 mean / median','Pooled F2 mean / median',
           'Pooled F3 mJ mean / median','F3 skew'],
          [[short[b],a,*[f"{r['pooled_mean']:.4f} / {r['pooled_median']:.4f}"
            for label in ['F1','F2','F3'] for r in aggregates
            if (r['benchmark'],r['arm'],r['outcome'])==(b,a,label)],
            *[f(r['pooled_skew'],2) for r in aggregates
              if (r['benchmark'],r['arm'],r['outcome'])==(b,a,'F3')]] for b in bench for a in arms])
    para('### 5.1 Actual policies beating the baseline mean')
    para('Shares below pool policies and use strict improvement over the repeated-default mean. Different policies can contribute to different marginal shares; the final column counts policies improving **all three objectives simultaneously**, relative to that baseline mean vector. These are descriptive comparisons without a repeatability guarantee for any one policy.')
    jmap={(r['benchmark'],r['arm']):r for r in joint}
    table(['Benchmark','Arm','F1','F2','F3','Feasibility','All F1/F2/F3'],
          [[short[b],a,*[f"{ns['share_tbl'][(b,o,a)]:.1f}%" for o in ['F1','F2','F3','feasibility']],
            f"{jmap[(b,a)]['count']}/{jmap[(b,a)]['total']} ({jmap[(b,a)]['percent']:.1f}%)"] for b in bench for a in arms])
    fig('share_beating_default','Per-objective shares of retained policies strictly beating the default mean.')
    para('### 5.2 Whole-front tests against default')
    para('Each repetition is summarised by its front mean or coordinate-wise median and compared with the five default repeats. Holm adjusts four outcomes within each benchmark-arm cell. The table lists every corrected survivor; full results are in the JSON and notebook.')
    table(['Estimator','Nominal p < .05','Holm(4) p < .05'],[[name,sum(r['p']<.05 for r in rows),sum(r['p_holm']<.05 for r in rows)] for name,rows in whole.items()])
    table(['Estimator','Benchmark','Arm','Outcome','Favourable margin','Nominal p','Holm(4) p'],
          [[name,short[r['benchmark']],r['arm'],r['outcome'],f"{r['margin']:+.4f}",f(r['p']),f(r['p_holm'])] for name,rows in whole.items() for r in rows if r['p_holm']<.05])
    para('''The previous blanket “no whole-front significance” claim was incorrect.
However, no identical benchmark/arm/outcome survives under **both** mean and
median aggregation. Mean–median disagreements, particularly for C energy,
make a summary-estimator-independent superiority claim unsupported.''')
    para('## 6. Between-run variance and cross-arm contrasts')
    para('SD is across five per-repetition front means; CV = SD / |mean|. In particular, the MOBO energy SD uses all five repetitions. Adaptive search could propagate measurement noise, but these SDs do not isolate that mechanism from other seed and run variation.')
    table(['Benchmark','Arm','F2 SD (CV)','F1 SD (CV)','F3 SD (CV)'],
          [[short[b],a,*[f"{r['sd']:.4f} ({r['cv_pct']:.2f}%)" for label in ['F2','F1','F3'] for r in aggregates if (r['benchmark'],r['arm'],r['outcome'])==(b,a,label)]] for b in bench for a in arms])
    para('Cross-arm MWU p-values below compare per-repetition front means. Holm(3) corrects the three objectives within each benchmark/arm pair. The sign of Δ and δ is Y − X, so positive favours Y only for F2.')
    table(['Benchmark','Y − X','Outcome','Δ','Cliff δ','Nominal p','Holm(3) p'],
          [[short[r['benchmark']],f"{r['arm_y']} − {r['arm_x']}",r['objective_key'].removeprefix('mean_'),f"{r['diff']:+.4f}",f"{r['delta']:+.3f}",f(r['p_rep']),f(r['p_holm'])] for r in cross_arm])
    para('Only C F2, RS versus GA, survives this three-outcome correction. There are twelve MOBO-involving objective contrasts across two benchmarks; none rejects at nominal .05. This is not evidence that the arms are equivalent.')
    para('## 7. Set-level quality indicators')
    para('Objectives are mapped to minimisation (F1, −F2, F3), then normalised using the pooled benchmark-wide ideal and range. Hypervolume retains the original 200,000 uniform Monte Carlo points and seed 20260825, with reference (1.10, 1.10, 1.10). Cached domination masks accelerate the calculation without changing the estimator. Values are estimates; the displayed precision does not remove Monte Carlo error.')
    table(['Benchmark','Arm','Raw HV mean ± SD','HV at 33 points mean ± SD','IGD⁺ mean ± SD','ε⁺ mean ± SD','Union contribution'],
          [[short[b],a,pm(ns['hv_vals'][(b,a)]),pm(ns['SIZEMATCH'][(b,a)]),pm(ns['IND'][(b,a)]['igd']),pm(ns['IND'][(b,a)]['eps']),f"{ns['IND'][(b,a)]['contrib']}/{ns['IND'][(b,a)]['ref']}"] for b in bench for a in arms])
    para('The size control draws 33 policies without replacement, forty times per front, and averages the resulting HV within each repetition. Its SD is between repetitions, not over all resamples. The union reference for IGD⁺ and ε⁺ is empirical, not the unknown true Pareto front. All three arms contribute non-dominated points.')
    table(['Benchmark','Pair','Raw HV nominal p','Controlled HV nominal p'],
          [[short[b],f'{x}–{y}',f(between(ns['hv_vals'][(b,x)],ns['hv_vals'][(b,y)])[0]),f(between(ns['SIZEMATCH'][(b,x)],ns['SIZEMATCH'][(b,y)])[0])] for b in bench for x,y in combinations(arms,2)])
    para('No raw or cardinality-controlled HV comparison reaches nominal .05. The mean-HV leading arm stays MOBO on C and GA on Python across reference margins 1.02–1.50 and after cardinality control. A stable leading arm is descriptive, not a statistically established ordering.')
    fig('set_indicators','Held-out set indicators across all five repetitions of every arm.')
    para('## 8. Paired upstream-task inference')
    para('The corrected bootstrap resamples 98 C tasks and 68 Python tasks, not 100 independent C records. Observations are first pooled within task across policies/records; each task then has equal weight. F1/F3 use feasible observations only. Larger fronts contribute more observations within a task. Intervals condition on the saved policies and do not capture search-run uncertainty or repair the train/test task overlap. The table shows correctness contrasts; all objective contrasts are exported.')
    table(['Benchmark','Contrast','Tasks','Δ F2','95% task-bootstrap CI','CI excludes 0','Wilcoxon nominal p'],
          [[short[b],f'{x} − {y}',r['n'],f"{r['mean']:+.4f}",f"[{r['lo']:+.6f}, {r['hi']:+.6f}]",str(r['excl']),f"{r['p']:.4g}"] for (b,x,y,k),r in ns['contrasts'].items() if k=='correctness'])
    fig('paired_bootstrap_f2','Paired task-bootstrap correctness differences; C grouping is corrected to 98 tasks.')
    para('## 9. Training-to-held-out generalisation')
    para('These are correlations among retained policies, pooled over repetitions. Nominal policy-level p-values do not account for within-repetition dependence; they are exploratory. Conditioning on the selected fronts restricts the population, but does not guarantee attenuation or make any coefficient a lower bound.')
    table(['Benchmark','Arm','F2 ρ (nominal p)','F1 ρ (nominal p)','F3 ρ (nominal p)'],
          [[short[b],a,*[f"{ns['TRAIN_RANK'][(b,a,k)]['rho']:+.3f} ({ns['TRAIN_RANK'][(b,a,k)]['p']:.3g})" for k in ['correctness','cl','energy']]] for b in bench for a in arms])
    para('Correctness has consistently positive transfer. C complexity also has positive associations, including MOBO; “only correctness transfers” is too strong. MOBO Python energy has a small positive nominal association (about ρ = .128, p = .046), so “all energy p-values exceed .12” was false. It is weak evidence after dependence and multiplicity are considered.')
    fig('train_vs_heldout','Per-policy training versus held-out correctness, including all five MOBO repetitions.')
    para('## 10. Policy selection')
    para('### 10.1 Training-selected shortlist versus held-out oracle')
    h=exported['harmonic_selection']
    para(f"Operational shortlists rank policies by recorded training fitness and average the five selected policies' held-out outcomes within each repetition. Harmonic training versus held-out top-five overlap is {h['mean_overlap']:.2f}/5, versus a combinatorial random expectation of {h['chance_overlap']:.2f}/5; {h['zero_overlap']}/{h['cells']} cells have zero overlap. Mean within-front harmonic rank correlation is {h['mean_rho']:+.3f}, negative in {h['negative_rho']}/{h['cells']} cells. The overlap difference from chance is descriptive, not a reported significance test.")
    table(['Benchmark','Arm','Selected median held-out harmonic rank','Median front size','Aggregate F2 random-shortlist percentile'],
          [[short[b],a,f(r['median_heldout_harmonic_rank'],1),f(r['median_front_size'],1),f(r['percentile'],2)] for (b,a),r in ns['RANDOM_SELECTION'].items()])
    para('The random reference uses 2,000 independent five-without-replacement shortlists per repetition (seed 20260825), then averages the five repetition means for each draw. Percentiles refer to that aggregate distribution, not to each individual repetition. A low percentile describes this conditional selection reference, not an independently tested search arm.')
    agree=ns['TRAIN_AGREEMENT'][('harmonic','dist-to-ideal')]
    oracle=ns['ORACLE_AGREEMENT'][('harmonic','ideal')]
    para(f"Harmonic and ideal-distance **training top-five** sets are identical in {agree['identical']}/{agree['cells']} benchmark/arm/repetition cells, with mean overlap {agree['mean_overlap']:.3f}/5. The previously quoted {oracle}/{ns['N_CELLS']} agreement applies instead to a **single held-out-oracle policy**, using benchmark-wide normalisation. It does not describe the operational shortlist. The 30 training-selection cells repeat the same 15 search fronts across two benchmarks (8/15 distinct fronts agree). Agreement now compares exact policy vectors, not rounded outcome triples. No agreement count makes the rules preference-free.")
    # Plot the actual training-shortlist agreement separately from the oracle.
    rules = ns['RULES2']
    agreement = np.eye(len(rules))
    for i,x in enumerate(rules):
        for j,y in enumerate(rules):
            if i < j:
                r = ns['TRAIN_AGREEMENT'][(x,y)]
                agreement[i,j] = agreement[j,i] = r['identical']/r['cells']
    plt = ns['plt']
    image, ax = plt.subplots(figsize=(6.2,5.0))
    im = ax.imshow(agreement,vmin=0,vmax=1,cmap='Blues')
    labels = ['F2 only','harmonic','ideal distance','hyperplane','trade-off']
    ax.set_xticks(range(len(rules)),labels,rotation=35,ha='right')
    ax.set_yticks(range(len(rules)),labels)
    for i in range(len(rules)):
        for j in range(len(rules)):
            ax.text(j,i,f'{100*agreement[i,j]:.0f}%',ha='center',va='center',
                    color='white' if agreement[i,j]>.55 else 'black')
    ax.set_title('Training top-five agreement: identical policy sets\n15 search fronts (repeated on two benchmarks)')
    image.colorbar(im,ax=ax,fraction=.045)
    image.tight_layout(); ns['savefig'](image,'training_rule_agreement'); plt.show()
    fig('training_rule_agreement','Agreement of actual training-selected top-five sets; distinct from held-out oracle agreement.')
    para('### 10.2 Correctness-only training selection')
    para('The table reports the mean of five per-repetition top-five shortlists. Favourable margins are baseline − selected for F1/F3 and selected − baseline for F2/feasibility. F2 margins are **fractions**: multiply by 100 for percentage points. Relative percentage improvement is a different quantity.')
    table(['Benchmark','Arm','Selected F1','Selected F2','Selected F3 mJ','Selected feasible /100','F2 gain (pp)','F2 gain (relative %)'],
          [[short[b],a,*[f(ns['SELECT_TESTS'][('correctness only',b,a,o)]['selected'],4 if o!='feasible' else 2) for o,_,_ in keys],
            f(100*ns['SELECT_TESTS'][('correctness only',b,a,'F2')]['margin'],2),f(ns['SELECT_TESTS'][('correctness only',b,a,'F2')]['relative_pct'],2)] for b in bench for a in arms])
    table(['Benchmark','Arm','Outcome','Favourable margin','Nominal p','Holm(4) p','Holm(24) p'],
          [[short[r['benchmark']],r['arm'],r['outcome'],f"{r['margin']:+.4f}",f(r['p']),f(r['p_holm']),f(r['p_holm_global24'])] for r in selected])
    pos=sum(r['margin']>0 for r in selected);nom=sum(r['p']<.05 for r in selected);adj=sum(r['p_holm']<.05 for r in selected)
    para(f"There are **{pos}/24 positive margins**, not 24/24; the four negative margins are energy outcomes. {nom}/24 tests reject nominally and **{adj}/24 survive Holm within each four-outcome benchmark-arm family**, forming five F2/feasibility pairs. Python GA does not survive the within-cell correction. None survives correction over all 24 tests (minimum adjusted p={min(r['p_holm_global24'] for r in selected):.4f}). F2 and feasibility are mechanically linked, so these are not ten independent confirmations.")
    para('Correctness-only gives the largest mean F2 and feasibility among the five tested rules in all six cells. Its preference was informed by the held-out results; these findings are exploratory. The shortlist mean is not the outcome of a specified deterministic one-policy deployment rule. Complexity advantages of other rules vary by benchmark and arm; energy savings are not established.')
    table(['Training rule','Tests','Nominal survivors','Holm(4) survivors','Corrected outcomes'],
          [[rule,len(rr),sum(r['p']<.05 for r in rr),sum(r['p_holm']<.05 for r in rr),'; '.join(f"{short[r['benchmark']]}/{r['arm']}/{r['outcome']}" for r in rr if r['p_holm']<.05) or 'None'] for rule in ns['RULES2'] for rr in [[r for r in selection if r['rule']==rule]]])
    fig('selection_basis_margins','Shortlist margins by selection basis and rule; held-out selection is an oracle diagnostic.')
    para('### 10.3 C sensitivity excluding the seven records on training tasks')
    para('This post-hoc sensitivity retains the original training-selected policy shortlists and recomputes their outcomes and the repeated baseline on the remaining **93 records from 92 tasks**. It changes the evaluation subset, not policy selection. Feasibility is expressed as a percentage of 93 for comparability; its margin is in percentage points. This diagnostic does not make the original 100-record experiment fully task-disjoint.')
    table(['Arm','Outcome','Baseline','Selected','Favourable margin','Nominal p','Holm(4) p'],
          [[r['arm'],r['outcome'],f(r['baseline']),f(r['selected']),f"{r['margin']:+.4f}",f(r['p']),f(r['p_holm'])] for r in sensitivity])
    survivors = [f"{r['arm']}/{r['outcome']}" for r in sensitivity if r['p_holm'] < .05]
    para('All three arms retain positive mean correctness margins on the reduced subset. '
         'After within-arm Holm(4), the surviving outcomes are: ' + ', '.join(survivors) +
         '. The loss of significance for GA and RS is a sensitivity of this small design; '
         'it does not by itself identify how much of the original effect was caused by task overlap.')
    para('## 11. Anytime convergence')
    para('All curves concern cumulative **training** hypervolume at checkpoints 50, 100, …, 750. RS and MOBO log every candidate; GA logs surviving populations and omits rejected offspring. GA absolute archive HV is a lower bound on all evaluated candidates, but the direction of bias after dividing by its own final HV is unknown. Comparisons involving GA are therefore descriptive archive comparisons.')
    table(['Arm','90% of own final HV','95%','99%','Normalised AUC','Last >0.5% gain checkpoint'],
          [[a,*[pm([ns['reach'][(a,i)][k] for i in ns['traj_reps'](a)],0) for k in range(3)],pm([ns['auc'][(a,i)] for i in ns['traj_reps'](a)]),pm([ns['plat'][(a,i)] for i in ns['traj_reps'](a)],0)] for a in arms])
    table(['RS–MOBO outcome','Nominal MWU p','Holm(5) p'],[[name,f(r['p'],5),f(r['p_holm'],5)] for (x,y,name),r in ns['ANYTIME_TESTS'].items() if (x,y)==('RS','MOBO')])
    para('''RS attains fractions of its own final training HV earlier on average. Of
the five RS–MOBO anytime outcomes, **only normalised AUC survives Holm(5)**.
Threshold times and the plateau have nominal evidence but do not survive that
family correction. These self-final ratios do not compare a common absolute
quality target. Non-significant final held-out differences do not establish
equal quality or equal-outcome cost dominance.

The plateau is the first checkpoint after the last individual gain exceeding
0.5% of final HV. Later smaller gains may accumulate. It is a retrospective
summary, not a validated stopping rule, a direct measure of wasted budget, or a
measured saving in wall time or energy. GA's incomplete archive prevents an
unqualified GA–RS efficiency claim.''')
    fig('anytime_compare','Anytime training summaries: GA survivor snapshots; RS and MOBO full evaluation logs.')
    para('## 12. Cross-language behaviour and descriptive efficiency')
    cross=ns['CROSS_LANGUAGE']
    para(f"All {cross['n']} policy vectors match exactly across benchmarks. Pooled C-versus-Python F2 Spearman ρ = {cross['rho']:+.3f} (nominal p={cross['p']:.3g}). Pooled F2 SD is {cross['sd_c']:.4f} on C and {cross['sd_py']:.4f} on Python. These associations condition on retained policies; narrower spread does not make the observed correlation a lower bound.")
    table(['Arm','Within-arm cross-language F2 ρ'],[[a,f(spearmanr([p[1] for p in ns['pairs'] if p[0]==a],[p[2] for p in ns['pairs'] if p[0]==a]).statistic,3)] for a in arms])
    table(['Benchmark','Arm','Mean of per-policy F2 / execution-energy (J) ratios'],[[short[r['benchmark']],r['arm'],f(r['ratio'],3)] for r in efficiency])
    para('Each ratio divides a policy’s mean F2 by its mean feasible-record energy in joules; the table averages those ratios across policies. The baseline averages its five repetition-level ratios. This is a mean of ratios, not a ratio of pooled means. The numerator and denominator have different feasibility conditioning. It is not a per-task efficiency estimate at fixed correctness, nor does it include inference/search cost or establish a break-even deployment count.')
    para('## 13. Per-task structure')
    table(['Benchmark','Tasks','RS > GA','RS < GA','Tied','No passing test in any observation','Every observation fully correct'],
          [[short[b],r['n'],r['rs_better'],r['rs_worse'],r['tied'],r['no_passing_tests'],r['every_observation_fully_correct']] for b,r in ns['PROBLEM_SUMMARY'].items()])
    para('“No passing test” means F2 = 0 across every observed policy and baseline attempt on that task. “Every observation fully correct” requires F2 = 1 throughout. These are observed extremes, not estimates of an intrinsic achievable repair ceiling. In particular, feasibility is partial test success and must not be described as full repair success.')
    para('## 14. Statistical sensitivity and front geometry')
    para('The notebook retains repetition-level Cliff’s δ bootstrap intervals, exhaustive permutation tests of mean differences, a normal-model power simulation, exploratory Welch TOST, Shapiro diagnostics, ECDF/QQ plots, Kruskal–Wallis/Dunn tests, front contributions, objective correlations and policy-space PCA. Their roles and limitations differ:')
    para('''- Bootstrap effect intervals with five observations per arm are uncertain;
  they do not create additional independent repetitions.
- Exact enumeration of label permutations removes Monte Carlo error in that
  permutation test, conditional on exchangeability. It need not equal MWU because
  it tests a different statistic.
- The normal-model power simulation uses 3,000 draws per effect size and nominal
  α=.05. Its approximate detectable effect is model- and variance-dependent;
  it is not empirical power for every benchmark outcome or a corrected family.
- TOST uses an estimated one-baseline-SD margin and uncorrected Welch tests.
  Practical equivalence depends on accepting that margin and those assumptions.
  Non-significant MWU alone is not equivalence.
- Non-rejection of Shapiro with five observations does not validate normality.
- Omnibus and pairwise tests reuse the same data; they are corroborating analyses,
  not independent replications. Rep index does not define a justified cross-arm
  block for a Friedman test.''')
    tally=Counter(v[4] for v in ns['TOSTR'].values())
    table(['Exploratory TOST/MWU classification','Count /18'],[[k,v] for k,v in sorted(tally.items())])
    table(['Benchmark','Outcome','Kruskal–Wallis H','Nominal p','Mean ranks GA / RS / MOBO'],
          [[short[b],key.removeprefix('mean_'),f(v[0],3),f(v[1]),' / '.join(f(x,1) for x in v[2])] for (b,key),v in ns['KW'].items()])
    for (b,key),(h,p,_,_) in ns['KW'].items():
        if p < .05:
            sign=-1 if key=='mean_correctness' else 1
            res=ns['dunn']([sign*np.array(per_rep(b,a,key)) for a in arms],arms)
            table([f'Dunn post-hoc: {short[b]} {key}','Nominal p','Holm(3) p'],[[label,f(raw),f(adj)] for (label,_,raw),adj in zip(res,holm([r[2] for r in res]))])
    para(f"The first two policy-space principal components explain {100*ns['evr'].sum():.1f}% of standardised 11-dimensional variance. Overlap in that projection does not establish full-space inseparability or isotropy. F2 and feasibility are mechanically linked; their correlation is not independent evidence for two effects. All arms contribute to the pooled non-dominated sets, so no arm covers every useful region.")
    para('## 15. Findings suitable for reporting')
    para('''1. The completed comparison comprises five repetitions per arm, 222 GA,
   198 RS and 245 MOBO policies, evaluated on each benchmark.
2. Whole-front mean and median tests give some corrected improvements against
   default, but the surviving outcomes are estimator-dependent. A blanket null
   result is incorrect; a universal front-level advantage is also unsupported.
3. Training-correctness top-five shortlists improve mean F2 and feasibility in
   all six cells. Ten outcomes survive four-outcome within-cell Holm correction;
   none survives global correction across 24. The rule was favoured post hoc.
4. MOBO’s five-repetition numbers must replace the older three-repetition
   variance, indicator, correlation and selection entries. No HV comparison is
   significant at nominal .05 after including all repetitions.
5. RS–MOBO normalised training AUC survives correction over five anytime
   summaries. This does not establish equal held-out quality, validated budget
   reductions or end-to-end cost savings.
6. The C split has six overlapping upstream tasks. Report record-level held-out
   performance honestly and use the separate 93-record sensitivity only with its
   changed denominator and post-hoc status stated.''')
    para('## 16. Threats to validity')
    para('''Five repetitions, one model, one held-out sample per benchmark and one
language pair limit precision and external validity. Search and evaluation hosts
differ; held-out real RAPL measurements on a common host improve comparability
but do not rule out drift or repair stochasticity. Each retained policy was
measured once, while only default was repeated. Training energy remains part of
training-based selection and the anytime criterion.

Conditional F1/F3 can change when the feasible task composition changes. Front
cardinality affects pooled weighting and HV; the subsampling control is a
sensitivity analysis. Task bootstrap intervals condition on the retained
policies. C task overlap persists in the original results. Choice of selection
rule, shortlist size, multiplicity family and indicator can affect conclusions.
Default decoding is not an expert-tuned comparator. Incomplete GA histories,
self-final normalisation and retrospective plateau definitions limit cost claims.''')
    para('## 17. Reproduction and figure replacement')
    para('''Regenerate the notebook from
`scripts/generators/build_three_arm_notebook.py`, then
execute every cell from the repository root.
The final analysis cell regenerates this report and
`analysis/data/three_arm_results.json`. Raw held-out artefacts under `outputs/`
are required in addition to committed training fronts and histories; the JSON
records their input hashes. No experiment is rerun by this analysis.

Figures are written to `analysis/figs/` as `fig_*.png` by the notebook and are
embedded from there.''')
    data_path.write_text(json.dumps(_jsonable(exported),indent=2,allow_nan=False)+'\n')
    report_path=repo/'docs'/'reports'/'replicated_three_arm_comparison.md'
    report_path.write_text('\n\n'.join(out)+'\n')
    print(f'  wrote {data_path.relative_to(repo)}')
    print(f'  wrote {report_path.relative_to(repo)}')
    print('  C unseen-task sensitivity:')
    for r in sensitivity:
        print(f"    {r['arm']} {r['outcome']}: margin {r['margin']:+.4f}, Holm(4) p={r['p_holm']:.5f}")


def baseline_summary(baseline):
    return {b:dict(summary=v['summary'], per_rep=[{k:r[k] for k in
        ['mean_cl','mean_correctness','mean_energy','n_feasible','n_total']} for r in v['reps']])
        for b,v in baseline.items()}
