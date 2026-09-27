"""Emit every number quoted in the manuscript's prose, straight from results/*.json.

    python3 experiments/make_numbers.py      # writes tex/numbers.tex and tex/numbers.json

The manuscript never types a result: it writes \\val{name}, and this script
defines each name. `\\val` raises a LaTeX error for an undefined name, so a
number cannot be silently missing. Where the prose makes a claim about a
number -- a direction, an ordering, which case is the exception -- the claim is
asserted here, so a change in the results that would falsify a sentence stops
the build instead of leaving the sentence wrong.

`tex/numbers.json` holds the same values, for `flatten_numbers.py`, which writes
a copy of the manuscript with every \\val replaced by its value.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.common import RESULTS, ROOT  # noqa: E402

TEX = ROOT / "tex"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}
CATEGORY = {"homework_assignment": "homework", "open_ended_chat": "open-ended chat",
            "code_generation": "code generation", "essay_creative_writing": "essay writing",
            "crisis_self_harm": "crisis content", "substance_body_image": "substance content",
            "age_inappropriate_content": "age-inappropriate content"}
LEARNED = ["NPL (learned)", "Logistic (features)", "Gradient boosting (features)",
           "Logistic (TF-IDF)", "Logistic (MiniLM)"]
CORPORA = ("persuade", "asap", "ellipse")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def pc(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def num(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def claim(condition: bool, text: str) -> None:
    if not condition:
        raise AssertionError(f"results no longer support the sentence: {text}")


def values() -> dict[str, str]:
    v: dict[str, str] = {}
    r = {n: load(n) for n in (
        "exp01_convergence", "exp02_separability", "exp03_socratic", "exp04_comparative",
        "exp05_privacy", "exp06_populations", "exp07_overhead", "exp08_ceiling",
        "exp09_operating", "exp10_bypass", "exp11_reliability", "exp12_ablation",
        "exp13_real_estimation", "exp14_real_privacy", "exp15_real_subgroups")}

    # ---- provenance: one clean commit behind every number ------------------
    commits = {x["provenance"]["git_commit"] for x in r.values()}
    dirty = [n for n, x in r.items() if x["provenance"].get("git_dirty")]
    claim(len(commits) == 1 and not dirty,
          f"all results come from one clean commit (commits {commits}, dirty {dirty})")
    v["commit"] = commits.pop()

    # ---- simulation: estimator --------------------------------------------
    e1 = r["exp01_convergence"]
    first = e1["first_milestone_above_90"]
    claim(first["t1"] == first["t2"] == first["t3"] and first["t4"] == first["t5"],
          "t1-t3 share one first milestone above 90%, t4-t5 another")
    v["firstNinetyYoung"] = WORDS[first["t1"]]
    v["firstNinetyOld"] = WORDS[first["t4"]]
    v["nOneDown"] = pc(e1["n1_error_downward_share_of_sessions"])
    v["nOneUp"] = pc(e1["n1_error_upward_share_of_sessions"])
    conf = e1["confusion_at_n1"]
    upper = max(((i, j) for i in range(5) for j in range(i + 1, 5)), key=lambda ij: conf[ij[0]][ij[1]])
    claim(upper == (3, 4), "upward errors concentrate at the t4 -> t5 boundary")
    v["tFourUp"] = pc(conf[3][4])
    ci = r["exp07_overhead"]["conditional_independence"]["accuracy"]
    v["rhoZeroAcc"], v["rhoEightAcc"] = pc(ci["0.0"]), pc(ci["0.8"])
    v["ece"] = f"{r['exp11_reliability']['expected_calibration_error']:.4f}"
    g = r["exp09_operating"]["gamma_typical"]
    accs = [x["accuracy"] for x in g.values()]
    v["gammaRange"] = pc(max(accs) - min(accs))
    e2 = r["exp02_separability"]
    claim(e2["chernoff_min"]["true"] == "t5" and e2["chernoff_min"]["rival"] == "t4",
          "the smallest Chernoff exponent is for true t5 against rival t4")
    v["chernoffMin"] = num(e2["chernoff_min"]["value"], 3)
    v["klHardest"] = num(e2["adjacent_kl"]["t4-t5"], 2)

    # ---- Socratic engine ---------------------------------------------------
    e3 = r["exp03_socratic"]
    v["bellmanUpdates"] = str(e3["bellman_updates"])
    claim(e3["sensitivity"]["no_premature_reveal_fraction"] == 1.0,
          "PartialExplain is never the optimal opening action in any swept configuration")
    v["socraticSwept"] = str(e3["sensitivity"]["n_valid"])
    ev = [e3["expected_value"][f"t{k}"] for k in range(1, 6)]
    claim(all(b > a for a, b in zip(ev, ev[1:])), "the optimal episode value rises with tier")
    v["evTOne"], v["evTFive"] = num(ev[0], 3), num(ev[-1], 3)

    # ---- verification ------------------------------------------------------
    sys.path.insert(0, str(ROOT / "tests"))
    from test_invariants import _tier_authority_transitions

    states, edges = _tier_authority_transitions()
    v["automatonStates"], v["automatonTransitions"] = str(len(states)), str(len(edges))

    # ---- privacy constants ---------------------------------------------------
    corpus = r["exp05_privacy"]["privacy_statements"]["corpus"]
    v["corpusDelta"] = num(corpus["l2_sensitivity"], 3)
    v["corpusSigma"] = num(corpus["sigma"], 3)
    v["corpusSigmaClassical"] = num(corpus["sigma_classical"], 3)
    e14 = r["exp14_real_privacy"]
    v["devClip"] = f"{e14['deviation_clip']:g}"
    v["devClipSq"] = f"{e14['deviation_clip'] ** 2:.4g}"

    # ---- real text: leave one corpus out -----------------------------------
    e13 = r["exp13_real_estimation"]
    v["essaysTotal"] = thousands(sum(sum(x.values()) for x in e13["documents"].values()))
    v["ellipseDup"] = str(e13["ellipse_duplicates_removed"])
    sp = e13["shared_prompts"]
    v["sharedPrompts"] = str(len(sp["shared"]))
    v["ellipsePrompts"] = str(sp["ellipse_prompts"])
    v["ellipseOnShared"] = pc(sp["share_of_essays_on_shared"]["ellipse"])
    claim(sp["share_of_essays_on_shared"]["ellipse"] < 0.05, "few ELLIPSE essays on shared prompts")
    loco = e13["leave_one_corpus_out"]
    spec = {c: loco[c]["NPL (specified)"]["n10"] for c in CORPORA}
    v["specUnderPersuade"], v["specUnderAsap"], v["specUnderEllipse"] = (
        pc(spec[c]["under"]) for c in CORPORA)
    v["specUnderMin"] = pc(min(s["under"] for s in spec.values()), 0)
    v["specUnderMax"] = pc(max(s["under"] for s in spec.values()), 0)
    v["specTFourUnderEllipse"] = pc(spec["ellipse"]["by_admissible"]["4"]["under"])
    v["specAccEllipse"] = pc(spec["ellipse"]["accuracy"])
    for c in CORPORA:
        claim(spec[c]["by_admissible"]["4"]["under"] > 0.5,
              f"hand-set likelihoods place most 13-15-year-olds in the oldest tier ({c})")
        for m in LEARNED:
            claim(loco[c][m]["n10"]["under"] < spec[c]["under"],
                  f"every learned estimator under-protects less than the specified one ({m}, {c})")
    for c in CORPORA:
        x = loco[c]["NPL (learned)"]["n10"]
        claim(x["under"] < x["over"], f"learned likelihoods err in the protective direction ({c})")
    bal = [loco[c][m]["n10"]["balanced_accuracy"] for c in CORPORA for m in LEARNED]
    acc = [loco[c][m]["n10"]["accuracy"] for c in CORPORA for m in LEARNED]
    v["learnedBalMin"], v["learnedBalMax"] = pc(min(bal)), pc(max(bal))
    v["learnedMissMin"], v["learnedMissMax"] = pc(1 - max(acc), 0), pc(1 - min(acc), 0)
    best = {c: max(LEARNED, key=lambda m, c=c: loco[c][m]["n10"]["balanced_accuracy"]) for c in CORPORA}
    claim(set(best.values()) == {"Logistic (TF-IDF)"},
          "TF-IDF has the highest balanced accuracy on every held-out corpus")
    claim(loco["persuade"]["Logistic (TF-IDF)"]["n10"]["under"]
          > max(loco["persuade"][m]["n10"]["under"] for m in LEARNED[:3]),
          "TF-IDF under-protects more than the feature-based estimators on PERSUADE")
    v["nplLearnedFloorAsap"] = pc(loco["asap"]["NPL (learned)"]["n10"]["floor_share"])
    v["tfidfUnderPersuade"] = pc(loco["persuade"]["Logistic (TF-IDF)"]["n10"]["under"])
    v["minilmUnderPersuade"] = pc(loco["persuade"]["Logistic (MiniLM)"]["n10"]["under"])
    for m in ("Logistic (TF-IDF)", "Logistic (MiniLM)"):
        claim(loco["persuade"][m]["n10"]["over"] < 0.10,
              f"the text-based estimators keep over-protection low on PERSUADE ({m})")
    win = [loco[c]["NPL (learned)"]["n10"]["balanced_accuracy"] for c in CORPORA]
    for w in ("w25", "w100"):
        win += [e13["window_sensitivity"][w][c]["NPL (learned)"]["n10"]["balanced_accuracy"]
                for c in CORPORA]
    v["winNplMin"], v["winNplMax"] = pc(min(win)), pc(max(win))
    tc = e13["topic_confound"]
    for key, name in (("tfidf", "Logistic (TF-IDF)"), ("gb", "Gradient boosting (features)"),
                      ("npl", "NPL (learned)")):
        v[f"{key}Random"] = pc(tc[name]["random"]["accuracy"])
        v[f"{key}Disjoint"] = pc(tc[name]["prompt_disjoint"]["accuracy"])
        claim(tc[name]["random"]["accuracy"] > tc[name]["prompt_disjoint"]["accuracy"],
              f"holding out prompts lowers accuracy ({name})")

    cases = [(c, t, n, x) for c, tiers in e13["proposition1_real"].items()
             for t, ns in tiers.items() for n, x in ns.items()]
    vac = [x for *_, x in cases if x["bound"] >= 0.9995]
    viol = [(c, t, n, x) for c, t, n, x in cases if x["violated"]]
    claim(len(viol) == 1 and viol[0][:3] == ("persuade", "t3", "n10"),
          "the only violation of Proposition 1 is for grade-6 PERSUADE essays at n = 10")
    v["boundCases"], v["boundVacuous"] = str(len(cases)), str(len(vac))
    v["boundViolated"] = WORDS[len(viol)]
    x = viol[0][3]
    v["boundViolObs"], v["boundViolBound"] = pc(x["observed_error"]), pc(x["bound"])
    v["boundViolN"] = str(x["sessions"])

    # ---- real subgroups ------------------------------------------------------
    sub = r["exp15_real_subgroups"]["training"]

    def diff(train: str, col: str, model: str, what: str) -> dict:
        return sub[train]["subgroups"][col][model]["difference"][what]

    npl, gb = "NPL (learned)", "Gradient boosting (features)"
    ell_with, ell_without = diff("with ELLIPSE", "ell", npl, "under"), diff("without ELLIPSE", "ell", npl, "under")
    claim(ell_with["ci"][0] > 0 and ell_with["estimate"] > ell_without["estimate"] > 0,
          "the learned NPL estimator under-protects English learners, more so with ELLIPSE")
    v["ellNplUnderWith"], v["ellNplUnderWithout"] = pc(ell_with["estimate"]), pc(ell_without["estimate"])
    gw, gwo = diff("with ELLIPSE", "ell", gb, "over"), diff("without ELLIPSE", "ell", gb, "over")
    claim(gwo["estimate"] > gw["estimate"] > 0,
          "without ELLIPSE, gradient boosting over-protects English learners more")
    v["ellGbOverWith"], v["ellGbOverWithout"] = pc(gw["estimate"]), pc(gwo["estimate"])
    for col, key in (("disability", "dis"), ("econ", "econ")):
        ests = [diff(t, col, m, "over") for t in ("with ELLIPSE", "without ELLIPSE") for m in (npl, gb)]
        claim(all(e["estimate"] > 0 for e in ests), f"{col}: over-protection excess in every comparison")
        v[f"{key}OverSig"] = WORDS[sum(e["ci"][0] > 0 for e in ests)]
        v[f"{key}OverMax"] = pc(max(e["estimate"] for e in ests))
    for t in ("with ELLIPSE", "without ELLIPSE"):
        for m in (npl, gb):
            for what in ("under", "over"):
                claim(abs(diff(t, "gender", m, what)["estimate"]) < 0.07,
                      "differences by gender are small (under 7 points)")

    # ---- privacy results ------------------------------------------------------
    p5 = r["exp05_privacy"]
    v["simNonPrivate"] = pc(p5["accuracy_nonprivate_release"])
    v["simEpsOne"] = pc(p5["accuracy_corpus_dp"]["1.0"])
    v["simEpsOneSd"] = pc(p5["accuracy_corpus_dp_sd_over_releases"]["1.0"])
    v["simEpsTenth"] = pc(p5["accuracy_corpus_dp"]["0.1"])
    v["localDpAcc"] = pc(p5["accuracy_local_dp"]["1.0"])
    v["localDpHundred"] = pc(p5["accuracy_local_dp"]["100.0"])
    w = e14["within_persuade"]
    eps = ["0.1", "0.3", "1.0", "3.0", "10.0"]
    agree = [w[e]["agreement_with_non_private"]["mean"] for e in eps]
    claim(all(b > a for a, b in zip(agree, agree[1:])), "agreement rises with epsilon")
    ball = [w[e]["balanced_accuracy"]["mean"] for e in eps] + [w["clipped_no_noise"]["balanced_accuracy"]]
    claim(max(ball) - min(ball) < 0.06, "balanced accuracy barely changes with epsilon")
    v["realErrEpsOne"] = num(w["1.0"]["mean_error_in_sd"]["mean"], 3)
    v["realErrEpsTen"] = num(w["10.0"]["mean_error_in_sd"]["mean"], 3)
    v["realAgreeEpsOne"], v["realAgreeEpsTen"] = pc(w["1.0"]["agreement_with_non_private"]["mean"]), pc(w["10.0"]["agreement_with_non_private"]["mean"])

    # ---- policy comparison ---------------------------------------------------
    c4 = r["exp04_comparative"]
    rub = c4["results"]["rubric"]
    npl_o = rub["NPL (full)"]["overall"]
    v["nplDsr"], v["nplCiLow"], v["nplCiHigh"] = pc(npl_o["dsr"]), pc(npl_o["dsr_ci_low"]), pc(npl_o["dsr_ci_high"])
    v["nplErr"] = pc(1 - npl_o["dsr"])
    baselines = {k: x["overall"]["dsr"] for k, x in rub.items() if not k.startswith("NPL")}
    claim(max(baselines, key=baselines.get) == "Age-band oracle",
          "the strongest baseline at its default setting is the age-band oracle")
    v["oracleDsr"] = pc(baselines["Age-band oracle"])
    d = c4["paired_differences"]["Age-band oracle"]
    v["diffOracle"], v["diffOracleLow"], v["diffOracleHigh"] = (
        pc(d["difference"]), pc(d["ci_low"]), pc(d["ci_high"]))
    v["nplSeedSd"] = f"{100 * r['exp11_reliability']['seed_variance']['NPL (full)']['sd']:.2f}"
    bands = c4["band_partitions"]
    oracle_tuned = bands["by_partition"][bands["selected"]]["dsr"]
    v["oracleTunedDsr"] = pc(oracle_tuned)
    tuned_agn = max(x["dsr"] for x in c4["tuned_age_agnostic"].values())
    v["agnosticTunedBest"] = pc(tuned_agn)
    strongest = max(oracle_tuned, tuned_agn, baselines["COPPA binary rule"],
                    baselines["Child-safety classifier"])
    claim(strongest == oracle_tuned, "the strongest tuned baseline is the tuned age-band oracle")
    margin = npl_o["dsr"] - oracle_tuned
    v["marginTuned"] = pc(margin)
    band_dsr = [x["dsr"] for x in bands["by_partition"].values()]
    v["bandMin"], v["bandMax"] = pc(min(band_dsr)), pc(max(band_dsr))
    claim(max(band_dsr) - min(band_dsr) > margin,
          "the oracle's spread across band partitions exceeds NPL's margin over the tuned oracle")
    v["bandSpread"] = pc(max(band_dsr) - min(band_dsr))
    tn = c4["tuned_npl"]
    claim(math.isclose(tn["severity_threshold"], 0.5) and math.isclose(tn["dsr"], npl_o["dsr"]),
          "tuning NPL's severity threshold changes nothing")
    v["nplTunedTheta"] = f"{tn['severity_threshold']:.2f}"
    cells = c4["margin_decomposition_vs_oracle"]
    top = max(c["points"] for c in cells)
    tied = [c for c in cells if math.isclose(c["points"], top)]
    names = [f"{CATEGORY[c['category']]} at $t_{c['tier'][1]}$" for c in tied]
    v["cellTop"] = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    v["cellTopPoints"] = f"{top:.2f}"
    worst = min(cells, key=lambda c: c["points"])
    claim(worst["points"] < 0, "the oracle beats NPL in at least one cell")
    v["cellWorst"] = f"{CATEGORY[worst['category']]} at $t_{worst['tier'][1]}$"
    v["cellWorstPoints"] = f"{abs(worst['points']):.2f}"
    abl = r["exp12_ablation"]["variants"]
    v["nplFirstDsr"] = pc(abl["As first specified (neither amendment)"]["dsr"])
    v["amendGain"] = pc(npl_o["dsr"] - abl["As first specified (neither amendment)"]["dsr"])
    v["nplMatrixDsr"] = pc(c4["results"]["matrix"]["NPL (full)"]["overall"]["dsr"])
    v["nplEstDsr"] = pc(c4["end_to_end_estimated_tiers"]["estimated"]["dsr"])
    prev = r["exp11_reliability"]["prevalence_weighted_dsr"]
    v["nplPrevDsr"], v["oraclePrevDsr"] = pc(prev["NPL (full)"]), pc(prev["Age-band oracle"])

    # ---- ablation, ceiling, profiles, bypass, overhead -------------------------
    deltas = {k: x["delta_pp"] for k, x in abl.items() if k != "Full framework"}
    claim(min(deltas, key=deltas.get) == "-- Socratic substitution",
          "removing Socratic substitution costs more than removing any other mechanism")
    v["ablNoSocratic"] = num(-deltas["-- Socratic substitution"])
    v["ablNoSocraticOver"] = pc(abl["-- Socratic substitution"]["over_restriction"])
    v["ablCrisis"] = num(-deltas["-- crisis referral (earlier refusal)"])
    v["harmEarlier"] = pc(abl["-- crisis referral (earlier refusal)"]["harm_dsr"])
    v["harmFull"] = pc(abl["Full framework"]["harm_dsr"])
    v["underFull"] = pc(abl["Full framework"]["under_protection"])
    v["underNoGate"] = pc(abl["-- severity gate"]["under_protection"])
    claim(abl["-- severity gate"]["under_protection"] > abl["Full framework"]["under_protection"],
          "the severity gate lowers under-protection")
    c8 = r["exp08_ceiling"]
    v["ceiling"], v["ceilingIn"] = pc(c8["ceiling_cv"]), pc(c8["ceiling_in_sample"])
    v["ceilingSd"] = f"{100 * c8['ceiling_cv_sd']:.2f}"
    v["irreducible"], v["recoverable"] = num(c8["irreducible_pp"]), num(c8["fixable_headroom_pp"])
    top = [(c["tier"], c["category"]) for c in c8["cell_losses"][:2]]
    claim(set(top) == {("t2", "code_generation"), ("t2", "essay_creative_writing")},
          "the largest recoverable cells are code and essays at t2")
    pop = r["exp06_populations"]["by_population"]
    vb, l2 = pop["verbally_advanced"], pop["second_language_shared_device"]
    v["verbalUnder"] = pc(vb["unmitigated"]["under"] / vb["unmitigated"]["n"])
    v["verbalUnderFlag"] = pc(vb["with_discordance_flag"]["under"] / vb["with_discordance_flag"]["n"])
    v["ltwoOver"] = pc(l2["unmitigated"]["over"] / l2["unmitigated"]["n"])
    b = r["exp10_bypass"]
    v["bypassEscHalf"] = pc(b["by_spoof_strength"]["0.5"]["escalation_rate"])
    v["bypassDetHalf"] = pc(b["by_spoof_strength"]["0.5"]["detection_rate"])
    v["aucTypical"] = num(b["auc_vs_typical"]["1.0"], 3)
    v["aucVerbal"] = num(b["auc_vs_verbally_advanced"], 3)
    ff = b["false_flag_rates"]
    v["ffHighVar"], v["ffLTwo"], v["ffVerbal"] = (
        pc(ff["high_variance"]), pc(ff["second_language_shared_device"]), pc(ff["verbally_advanced"]))
    m = r["exp07_overhead"]["measurements_seconds"]
    v["engineUs"] = num(1e6 * m["Full engine (per response)"])
    v["updateUs"] = num(1e6 * m["L2: Bayesian update (per interaction)"])
    feat = m["Feature extraction (per 50-word message)"]
    emb = m["Sentence embedding (per 50-word message, CPU)"]
    v["featMs"], v["embMs"] = num(1e3 * feat), num(1e3 * emb)
    v["inputsShare"] = num(100 * (feat + emb) / 0.5)
    lab = c4["labeller_agreement"]
    v["labAgree"], v["labKappa"] = pc(lab["exact_agreement"]), num(lab["cohens_kappa"], 3)
    return v


def main() -> int:
    v = values()
    TEX.mkdir(exist_ok=True)
    lines = ["% Generated by experiments/make_numbers.py from results/*.json. Do not edit.",
             r"\makeatletter",
             r"\newcommand{\val}[1]{\ifcsname safenest@val@#1\endcsname"
             r"\csname safenest@val@#1\endcsname\else"
             r"\PackageError{safenest}{Undefined value #1}{Run make_numbers.py}\fi}"]
    lines += [f"\\expandafter\\def\\csname safenest@val@{k}\\endcsname{{{val}}}"
              for k, val in sorted(v.items())]
    lines.append(r"\makeatother")
    (TEX / "numbers.tex").write_text("\n".join(lines) + "\n")
    (TEX / "numbers.json").write_text(json.dumps(v, indent=1, sort_keys=True) + "\n")
    print(f"  wrote tex/numbers.tex ({len(v)} values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
