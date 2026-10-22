"""Durable independent author/reviewer stages and replayable admission audits."""
import copy
from pathlib import Path
import threading
import time
import core as g
import batching as b
import inherited
from client import Client,StopCollection,MODEL,ENDPOINT,payload_for
from files import atomic


class AdmissionContext:
    """One scheduler's audited corpus; callers must still audit at run end.

    Construct only from records/index returned by audit_arm. The shared mapping
    and index are updated after each durable checkpoint decision, so concurrent
    jobs see earlier admissions without rereading the immutable ancestor chain.
    """
    def __init__(self, root, arm, records, index):
        self.root, self.arm = Path(root).resolve(), arm
        self.records, self.index = records, index
        self.thread_id = threading.get_ident()
        self.generation = 0

    def stage(self, root, cp, index):
        if (Path(root).resolve() != self.root or cp['arm'] != self.arm
                or index is not self.index or threading.get_ident() != self.thread_id):
            raise RuntimeError('admission_context_scope_changed')
        return _AdmissionStage(self, cp['id'])


class _AdmissionStage:
    """Check a decision group without changing the shared index before save."""
    def __init__(self, context, batch_id):
        self.context, self.batch_id = context, batch_id
        self.generation = context.generation
        self.local = g.DuplicateIndex()
        self.additions = []

    def check(self, episode):
        previous = self.context.records.get(episode['id'])
        if previous is not None:
            # A cached review of this job's already committed episode is an
            # exact replay, not a new candidate competing against itself.
            if previous != episode or previous['provenance']['batch_id'] != self.batch_id:
                raise RuntimeError('committed_admission_changed')
            return None
        self.context.index.check(episode)
        return copy.deepcopy(episode), self.local.check(episode)

    def add(self, checked):
        if checked is not None:
            episode, fingerprint = checked
            self.local.add(fingerprint)
            self.additions.append((episode, fingerprint))

    def commit(self):
        if (self.context.generation != self.generation
                or threading.get_ident() != self.context.thread_id):
            raise RuntimeError('admission_context_changed_during_commit')
        for episode, fingerprint in self.additions:
            if episode['id'] in self.context.records:
                raise RuntimeError('admission_id_already_committed')
            self.context.index.add(fingerprint)
            self.context.records[episode['id']] = episode
        self.context.generation += 1


def arm_root(root,arm):
    folder = root if arm=='production' else root/'benchmark'/arm
    for name in ('accepted','rejected','snapshots'): (folder/name).mkdir(parents=True,exist_ok=True)
    return folder


def materialize(root,cp):
    folder=arm_root(root,cp['arm'])
    for slot,outcome in cp.get('outcomes',{}).items():
        spec=cp['specs'][int(slot[1:])]
        record={'batch_id':cp['id'],'slot':slot,'arm':cp['arm'],'spec_id':spec['id'],
                'contract_sha256':cp['contract_sha256'],**outcome}
        if outcome['status']=='accepted':
            record['example_sha256']=g.digest(outcome['example'])
            path=folder/'accepted'/(spec['id']+'.json')
        else:
            path=folder/'rejected'/(spec['id']+'.json')
        if path.exists():
            if g.read_json(path)!=record: raise RuntimeError('existing_record_changed')
        else: atomic(path,record)


def annotated(ep,cp,slot,author,verifier,expected):
    ep=copy.deepcopy(ep)
    ep['reference_status']='ifm_author_membership_exact_blind_regroup_passed'
    ep['provenance'].update(api_endpoint=ENDPOINT,contract_sha256=g.digest(expected),
        policy_sha256=expected['source_sha256']['POLICY.md'],batch_id=cp['id'],batch_slot=slot,
        benchmark_arm=None if cp['arm']=='production' else cp['arm'],
        author_call_id=author['receipt']['call_id'],verifier_call_id=verifier['receipt']['call_id'],
        author_response_id=author['receipt']['response_id'],verifier_response_id=verifier['receipt']['response_id'],
        author_returned_model=author['receipt']['model'],verifier_returned_model=verifier['receipt']['model'],
        actual_author_batch_size=len(author['slots']),actual_verifier_batch_size=len(verifier['slots']),
        author_origin_revision=author.get('origin_revision',g.OUT.name),
        author_response_truncated=author['reply']['transport']['response_truncated'],
        verifier_response_truncated=verifier['reply']['transport']['response_truncated'],
        verifier_response_format=verifier['reply']['transport']['response_format'],
        verifier_identity_source=verifier['reply']['transport'].get('record_identity','explicit_case_id'),
        record_protocol='JSONL_author_with_independent_review_no_repair')
    return ep


def process_job(root,job,expected,index,client,stop,phase=lambda value:None,*,admission=None):
    """Serial driver retained for offline replay and the bounded legacy runner."""
    steps = process_job_steps(root,job,expected,index,client,stop,phase,admission=admission)
    result = None
    while True:
        try:
            request = steps.send(result)
        except StopIteration as done:
            return done.value
        try:
            result = client.chat(**request)
        except BaseException as exc:
            try:
                steps.throw(exc)
            finally:
                steps.close()
            raise


def process_job_steps(root,job,expected,index,client,stop,phase=lambda value:None,*,admission=None):
    """Yield network work; all admission decisions stay on the scheduler thread."""
    path=root/'batches'/(job['id']+'.json')
    cp=g.read_json(path) if path.exists() else {**job,'contract_sha256':g.digest(expected),
        'stages':{},'active_seconds':0,'run_id':client.run_id}
    if cp['contract_sha256']!=g.digest(expected) or any(cp[k]!=job[k] for k in ('id','arm','specs')):
        raise RuntimeError('checkpoint_contract_or_assignment_changed')
    if cp.get('completed'):
        materialize(root,cp)
        return cp
    if cp.get('halted'): raise StopCollection('saved_protocol_failure_requires_inspection')
    start=time.monotonic()
    slots={s['id']:f's{i}' for i,s in enumerate(cp['specs'])}
    by_slot={slots[s['id']]:s for s in cp['specs']}
    config=expected['batching']
    # On resume, exclude this checkpoint's own committed examples from the
    # comparison index, so cached author/reviewer requests keep the same members.
    def current_external_index():
        if admission is not None:
            return admission.stage(root,cp,index)
        external=g.DuplicateIndex()
        if cp['arm']=='production':
            for episode in inherited.records(root).values():
                external.add(external.check(episode))
        for accepted_path in arm_root(root,cp['arm']).joinpath('accepted').glob('*.json'):
            existing=g.read_json(accepted_path)
            if existing['batch_id']!=cp['id']:
                external.add(external.check(existing['example']))
        return external
    def save(_=None): atomic(path,cp)
    save()

    def call_stage(key,kind,members,request):
        stage=cp['stages'].setdefault(key,{'spec':{'id':cp['id']+':'+key},'kind':kind,'slots':members,
             'arm':cp['arm'],'batch_size':len(members)})
        if stage['slots']!=members or stage['kind']!=kind: raise RuntimeError('stage_members_changed')
        role=kind+'_records'
        stage[role+'_item_ids']=[by_slot[s]['id'] for s in members]
        turns,schema,budget=request
        if not stage.get('completed'):
            phase(kind+'_records')
            try:
                stage['reply'],stage['receipt']=yield {
                    'turns':turns,'schema':schema,'seed':by_slot[members[0]]['seed']+(kind=='verifier'),
                    'role':role,'candidate':stage,'checkpoint':save,'max_tokens':budget}
            except ValueError as exc:
                stage['failure']=b.error_code(exc)
                cp['halted']='protocol_failure_'+key
                save()
                raise StopCollection('protocol_failure_inspect_saved_response') from None
            stage['completed']=True
            save()
        if not stage['reply']['records'] and not (
                stage['reply']['transport'].get('model_output_failure') is True
                and set(stage['reply']['errors'])==set(members)):
            cp['halted']='no_readable_records_'+key
            save()
            raise StopCollection('no_readable_records_inspect_before_more_requests')
        return stage

    try:
        valid,errors,authors,reviewers={}, {}, {}, {}
        author_cases=1 if job['arm']=='b1' else config['author_cases']
        for number,chunk in enumerate(b.pack(cp['specs'],author_cases,config['author_tabs'])):
            stop()
            members=[slots[s['id']] for s in chunk]
            stage=yield from call_stage('a'+str(number),'author',members,b.author_request(chunk,slots))
            errors.update(stage['reply']['errors'])
            for slot,record in stage['reply']['records'].items():
                try:
                    ep=b.normalize_record(record,by_slot[slot],stage['receipt'])
                    # Keep reviewer membership stable across pause/resume.
                    # Concurrent corpus changes are checked at admission below.
                    valid[slot],authors[slot]=ep,stage
                except (ValueError,AssertionError) as exc: errors[slot]=b.error_code(exc)
        cp['author_errors']=dict(errors)
        cp['pending_verification']=list(valid)
        save()
        passed={}

        def commit_decisions():
            # Another job may have admitted an overlapping example during the
            # network wait. Recheck against the latest corpus before committing.
            next_index=current_external_index()
            outcomes={}
            for slot in by_slot:
                if slot in passed:
                    try:
                        ep=annotated(valid[slot],cp,slot,authors[slot],reviewers[slot],expected)
                        next_index.add(next_index.check(ep))
                        outcomes[slot]={'status':'accepted','example':ep,
                            'author_stage':next(k for k,v in cp['stages'].items() if v is authors[slot]),
                            'verifier_stage':next(k for k,v in cp['stages'].items() if v is reviewers[slot])}
                    except (ValueError,AssertionError) as exc: errors[slot]=b.error_code(exc)
                if slot not in outcomes and slot in errors:
                    outcomes[slot]={'status':'rejected','reason':errors[slot]}
                if slot in outcomes:
                    previous=cp.get('outcomes',{}).get(slot)
                    if previous:
                        assert {k:v for k,v in previous.items() if k!='decision_run_id'}==outcomes[slot], 'committed_decision_changed'
                        outcomes[slot]=previous
                    else: outcomes[slot]['decision_run_id']=client.run_id
            # Previously committed reviews may not have been replayed yet.
            # Keep them intact until their cached stages are visited again.
            cp['outcomes']={**cp.get('outcomes',{}),**outcomes}
            cp['pending_verification']=[slot for slot in valid if slot not in cp['outcomes']]
            cp['decision_run_id']=client.run_id
            save()
            if admission is not None:
                # The checkpoint is durable. Publish admissions in memory even
                # if materializing an export encounters a filesystem error.
                next_index.commit()
            materialize(root,cp)
            if admission is None:
                index.__dict__.update(next_index.__dict__)

        commit_decisions()
        for number,chunk in enumerate(b.pack(list(valid.values()),config['verifier_cases'],config['verifier_tabs'])):
            stop()
            selected={slots[ep['id']]:ep for ep in chunk}
            stage=yield from call_stage('v'+str(number),'verifier',list(selected),b.verifier_request(selected))
            errors.update(stage['reply']['errors'])
            for slot,record in stage['reply']['records'].items():
                try:
                    b.review_record(valid[slot],record)
                    passed[slot],reviewers[slot]=record,stage
                except (ValueError,AssertionError) as exc: errors[slot]=b.error_code(exc)
            commit_decisions()
            phase('review_committed')
        assert len(cp['outcomes'])==len(by_slot)
        cp.update(completed=True,completed_at=g.utc(),pending_verification=[])
        save()
        materialize(root,cp)
        return cp
    finally:
        cp['active_seconds']+=time.monotonic()-start
        save()


def verified_stage(root,cp,key,request):
    stage=cp['stages'][key]
    turns,descriptor,budget=request
    role=stage['kind']+'_records'
    seed=cp['specs'][int(stage['slots'][0][1:])]['seed']+(stage['kind']=='verifier')
    saved=g.read_json(root/'responses'/(stage['receipt']['call_id']+'.json'))
    assert saved['receipt']==stage['receipt']
    assert saved['result']['model']==stage['receipt']['model']==MODEL
    assert stage['receipt']['endpoint']==ENDPOINT
    if stage.get('origin_revision',g.OUT.name) != g.OUT.name:
        # Reuse only the pinned author reply. Its original prompt is retained;
        # new title rules revalidate the data without rewriting its provenance.
        assert stage['kind']=='author'
        origin=g.read_json(root/'source-origin.json')
        parent=Path(origin['parent_revision'])
        assert stage['origin_revision']==parent.name
        assert g.sha(parent/'contract.json')==origin['parent_contract_sha256']
        checkpoint_hash=origin.get('parent_checkpoints_sha256',{}).get(cp['id'],origin.get('parent_checkpoint_sha256'))
        assert checkpoint_hash and g.sha(parent/'batches'/(cp['id']+'.json'))==checkpoint_hash
        old_cp=g.read_json(parent/'batches'/(cp['id']+'.json'))
        assert all(cp[field]==old_cp[field] for field in ('id','arm','specs'))
        old_stage=old_cp['stages'][key]
        assert old_stage['completed'] and old_stage['kind']=='author'
        assert stage['slots']==old_stage['slots'] and stage['receipt']==old_stage['receipt']
        relative='responses/'+stage['receipt']['call_id']+'.json'
        pinned=next(item for item in origin['inherited_generation_responses'] if item['path']==relative)
        assert pinned['kind']=='author_records' and g.sha(root/relative)==pinned['sha256']
    else:
        assert saved['payload']==payload_for(turns,descriptor,seed,role,budget)
    assert g.digest(saved['payload'])==stage['receipt']['request_sha256']
    assert g.digest(saved['result'])==stage['receipt']['response_sha256']
    assert Client.parse(saved['result'],stage['receipt'],descriptor,role)[0]==stage['reply']
    return stage


def audit_arm(root,arm,expected):
    folder=arm_root(root,arm)
    specs={s['id']:s for s in g.specifications()}
    records=inherited.records(root) if arm=='production' else {}
    index,cache=g.DuplicateIndex(),{}
    for episode in records.values():
        index.add(index.check(episode))
    for path in sorted((folder/'accepted').glob('*.json')):
        record=g.read_json(path)
        cp=g.read_json(root/'batches'/(record['batch_id']+'.json'))
        assert cp.get('outcomes') and cp['arm']==arm and cp['contract_sha256']==g.digest(expected)
        assert all(s==specs[s['id']] for s in cp['specs'])
        slot=record['slot']
        outcome=cp['outcomes'][slot]
        assert outcome['status']=='accepted' and record['example']==outcome['example']
        assert record['contract_sha256']==g.digest(expected) and record['example_sha256']==g.digest(record['example'])
        if cp['id'] not in cache:
            slots={s['id']:f's{i}' for i,s in enumerate(cp['specs'])}
            episodes,authors={},{}
            allowed=[tuple(slots[s['id']] for s in chunk) for chunk in b.pack(cp['specs'],
                1 if arm=='b1' else expected['batching']['author_cases'],expected['batching']['author_tabs'])]
            for key,stage in cp['stages'].items():
                if stage['kind']!='author' or not stage.get('completed'): continue
                assert tuple(stage['slots']) in allowed
                chunk=[cp['specs'][int(s[1:])] for s in stage['slots']]
                verified_stage(root,cp,key,b.author_request(chunk,slots))
                for s,raw in stage['reply']['records'].items():
                    try:
                        episodes[s]=b.normalize_record(raw,cp['specs'][int(s[1:])],stage['receipt'])
                        authors[s]=stage
                    except (ValueError,AssertionError): pass
            for key,stage in cp['stages'].items():
                if stage['kind']!='verifier' or not stage.get('completed'): continue
                selected={s:episodes[s] for s in stage['slots']}
                assert len(selected)<=expected['batching']['verifier_cases']
                assert sum(ep['tab_count'] for ep in selected.values())<=expected['batching']['verifier_tabs']
                verified_stage(root,cp,key,b.verifier_request(selected))
            cache[cp['id']]=episodes,authors
        episodes,authors=cache[cp['id']]
        reviewer=cp['stages'][outcome['verifier_stage']]
        b.review_record(episodes[slot],reviewer['reply']['records'][slot])
        ep=annotated(episodes[slot],cp,slot,authors[slot],reviewer,expected)
        assert ep==record['example']
        assert ep['id'] not in records, 'inherited_example_id_reused'
        index.add(index.check(ep))
        records[ep['id']]=ep
    g.validate_corpus(list(records.values()))
    return records,index
