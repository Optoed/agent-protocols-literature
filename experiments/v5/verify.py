"""Recalculate V5 from the published records. Standard library only; no network calls."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
from fractions import Fraction
from decimal import Decimal
import zipfile

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rational(text):
    text = text.strip().replace("−", "-").replace(" ", "")
    match = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
    try:
        if match:
            return (-1 if match[1] == "-" else 1) * Fraction(int(match[2]), int(match[3]))
        if re.fullmatch(r"[+-]?\d+(?:\.\d+|/[+-]?\d+)?", text):
            return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def boxed(text):
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    start += 7
    depth = 0
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            if depth == 0:
                return text[start:end].strip()
            depth -= 1
    return None


def score(answer, gold, normalize=True):
    value = boxed(answer)
    if value is None:
        return 0
    if normalize:
        value = re.sub(r"^[A-Za-z]\s*=\s*", "", value).removeprefix(r"\$").strip()
        if "," in value:
            if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", value):
                return 0
            value = value.replace(",", "")
    expected = rational(gold)
    require(expected is not None, "Invalid rational reference")
    return int(rational(value) == expected)


def main():
    provenance = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    require(digest((ROOT / "records.zip").read_bytes()) == provenance["records_sha256"], "Archive hash mismatch")
    require(digest((ROOT / "plan.json").read_bytes()) == provenance["plan_sha256"], "Plan hash mismatch")
    for name, metadata in provenance["source_files"].items():
        require(digest((ROOT / "source" / name).read_bytes()) == metadata["sha256"], "Source hash mismatch: " + name)
    plan = json.loads((ROOT / "plan.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for phase in ("search", "confirmation", "test") for t in plan[phase]}
    require(len(tasks) == len({t['question'].strip() for t in tasks.values()}) == 80, "Task overlap")
    require([len(plan[p]) for p in ("search", "confirmation", "test")] == [24, 8, 48], "Split size mismatch")
    for phase in ("search", "confirmation", "test"):
        require(all(t['level'] == 'Level 5' for t in plan[phase]), 'Unexpected difficulty')
        counts = collections.Counter(t['domain'] for t in plan[phase])
        require(counts['algebra'] == counts['intermediate_algebra'] == len(plan[phase]) // 2, 'Domain imbalance')
    with zipfile.ZipFile(ROOT / "records.zip") as archive:
        manifest = json.loads(archive.read("SHA256.json"))
        require(set(archive.namelist()) == set(manifest) | {"SHA256.json"}, "Unexpected archive member")
        records = {}
        for name, checksum in manifest.items():
            raw = archive.read(name)
            require(digest(raw) == checksum, "Record hash mismatch: " + name)
            records[name] = json.loads(raw)
    calls = [d for name, d in records.items() if name.startswith("calls/")]
    interrupted = [d for name, d in records.items() if name.startswith("interrupted_attempts/calls/")]
    require(len(calls) == 3312 and len(interrupted) == 13, "Call count mismatch")
    require(len({d['response']['id'] for d in calls}) == len(calls), "Duplicate response IDs")
    require(len({d['identity'] for d in calls}) == len(calls), "Duplicate call identities")
    for d in calls:
        require(d['status'] == 'completed', 'Incomplete current call')
        require('calls/' + digest(d['identity'].encode()) + '.json' in records, 'Identity filename mismatch')
        require(d['cost'] == d['response']['usage']['cost'], 'Cost metadata mismatch')
        require(d['request']['model'] == d['response']['model'] == plan['model'], 'Model mismatch')
        require(d['request']['temperature'] == .2 and d['request']['max_tokens'] == 8192, 'Parameter mismatch')
    costs = {}
    for d in calls + interrupted:
        c = costs.setdefault(d['category'], {'completed': 0, 'unknown': 0, 'known_usd': Decimal('0')})
        c['completed'] += d['status'] == 'completed'
        if d.get('cost') is None:
            c['unknown'] += 1
        else:
            c['known_usd'] += Decimal(str(d['cost']))
    for cat, c in costs.items():
        stored = records['costs.json'][cat]
        require(abs(float(c['known_usd']) - stored['known_usd']) < 1e-10, 'Cost total mismatch')
        require(c['unknown'] == stored['unknown_cost_calls'], 'Unknown cost count mismatch')
    evaluations = {name: d for name, d in records.items() if name.startswith('evaluations/')}
    require(len(evaluations) == 1400, 'Evaluation count mismatch')
    for name, d in evaluations.items():
        require(score(d['answer'], tasks[d['id']]['answer']) == d['quality'], 'Scoring mismatch: ' + name)
        require(score(d['answer'], tasks[d['id']]['answer'], False) == d['strict_quality'], 'Strict scoring mismatch: ' + name)
    def rows(rep, phase, ids, fp):
        return [records[f'evaluations/{rep}-{phase}-{tid}-{fp}.json'] for tid in ids]
    def select(scores):
        return min(scores, key=lambda fp: (-scores[fp], len(plan['workflows'][fp]['nodes']), fp))
    correct = collections.defaultdict(list)
    domains = collections.defaultdict(lambda: collections.defaultdict(list))
    chosen = []
    for rep in range(plan['replicates']):
        primary = [t['id'] for t in plan['search']]
        reserve = [t['id'] for t in plan['confirmation']]
        scores = {phase: {fp: sum(d['quality'] for d in rows(rep, phase, ids, fp)) for fp in plan['workflows']}
                  for phase, ids in [('independent_search', primary + reserve), ('shared_search', primary)]}
        finalists = [fp for fp, value in scores['shared_search'].items() if max(scores['shared_search'].values()) - value <= plan['near_tie_margin']]
        confirmation = {fp: sum(d['quality'] for d in rows(rep, 'confirmation_search', reserve, fp)) for fp in finalists} if len(finalists) > 1 else {}
        combined = {fp: scores['shared_search'][fp] + confirmation.get(fp, 0) for fp in finalists}
        winners = {'independent': select(scores['independent_search']), 'shared': select(scores['shared_search']), 'shared_confirm': select(combined)}
        selection = records[f'selection-{rep}.json']
        for key, value in [('scores', scores), ('finalists', finalists), ('confirmation', confirmation), ('combined', combined), ('winners', winners)]:
            require(selection[key] == value, 'Selection mismatch: ' + key)
        require(records['result.json']['replicates'][rep] == records[f'replicate-{rep}.json'], 'Replicate/result mismatch')
        for mode, fp in winners.items():
            data = records[f'test-{rep}-{fp}.json']
            require(data == rows(rep, 'test_execution', [t['id'] for t in plan['test']], fp), 'Test rows mismatch')
            total = sum(d['quality'] for d in data)
            stored = records[f'replicate-{rep}.json']['test'][mode]
            require(total == stored['correct'] and len(data) == stored['count'] == 48, 'Test total mismatch')
            correct[mode].append(int(total))
            for domain in ['algebra', 'intermediate_algebra']:
                subtotal = sum(d['quality'] for d in data if d['domain'] == domain)
                require(subtotal == stored['by_domain'][domain]['correct'], 'Domain total mismatch')
                domains[mode][domain].append(int(subtotal))
        current = [d for d in calls if f'/rep-{rep}/' in d['identity']]
        last_search = max(datetime.datetime.fromisoformat(d['started_utc']) + datetime.timedelta(seconds=d['elapsed_seconds']) for d in current if d['category'] != 'test_execution')
        first_test = min(datetime.datetime.fromisoformat(d['started_utc']) for d in current if d['category'] == 'test_execution')
        require(last_search <= first_test, 'Test began before search finished')
        chosen.append({'replicate': rep, 'finalists': len(finalists), 'winners': {m: plan['labels'][fp] for m, fp in winners.items()}})
    saving = 1 - (costs['shared_search']['known_usd'] + costs['confirmation_search']['known_usd']) / costs['independent_search']['known_usd']
    summary = {'status': 'pass', 'completed_calls': len(calls), 'unknown_attempts': len(interrupted), 'evaluations': len(evaluations),
               'test_correct_out_of_48': dict(correct), 'mean_test_accuracy_percent': {m: sum(v) / 144 * 100 for m, v in correct.items()},
               'test_by_domain_out_of_24': dict(domains), 'search_known_saving_percent': float(saving * 100), 'costs': costs, 'selection': chosen,
               'scope': 'Published records, bounded rational scoring, selection, phase ordering, hashes and costs; no new model calls. Full engine replay and dataset sampling are described in audit.json.'}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
