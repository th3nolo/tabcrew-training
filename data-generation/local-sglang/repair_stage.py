"""One durable author repair before the first blind review; never retry a judge."""
import copy
import json
import core as g
import repair


def verify(saved, complete=False):
    item=saved.get('repair')
    if item is None:return
    original=item['original']
    if item['original_sha256']!=g.digest(original):raise RuntimeError('Repair original integrity mismatch')
    if item['reason'] not in repair.REPAIRABLE:raise RuntimeError('Ineligible repair reason')
    targets=repair.defects(original['authored'],saved['spec'])
    if not 1<=len(targets)<=8 or targets!=item['targets']:raise RuntimeError('Repair target mismatch')
    if 'receipt' in item:
        if item['replacement_sha256']!=g.digest(item['replacements']) or item['receipt_sha256']!=g.digest(item['receipt']):
            raise RuntimeError('Repair result integrity mismatch')
        parsed=json.loads(item['receipt']['raw_content'],object_pairs_hook=g.unique_object)
        if parsed!=item['replacements']:raise RuntimeError('Repair raw response mismatch')
        repair.apply(original['authored'],saved['spec'],targets,item['replacements'])
    if complete:
        if 'receipt' not in item:raise RuntimeError('Missing repair receipt')
        rebuilt=repair.apply(original['authored'],saved['spec'],targets,item['replacements'])
        if rebuilt!=saved['authored']:raise RuntimeError('Repaired content differs from original and patch')
        expected=g.digest([original['author_receipt']['request_sha256'],item['receipt']['request_sha256']])
        if saved['author_receipt']['request_sha256']!=expected:raise RuntimeError('Repair request provenance mismatch')


def normalize(saved,client,checkpoint,should_stop):
    spec=saved['spec']
    if 'repair' not in saved:
        try:
            episode=g.normalize(saved['authored'],spec,saved['author_receipt'])
            g.validate_corpus([episode])
            return episode
        except ValueError as error:
            if str(error) not in repair.REPAIRABLE or 'blind_review' in saved:raise
            targets=repair.defects(saved['authored'],spec)
            if not 1<=len(targets)<=8:raise
            original=copy.deepcopy({key:saved[key] for key in ('authored','author_receipt')})
            saved['repair']={'version':repair.VERSION,'reason':str(error),'targets':targets,
                'original':original,'original_sha256':g.digest(original)}
            saved['stage']='repair_pending';checkpoint(saved)
    verify(saved)
    item=saved['repair'];original=item['original']
    if 'receipt' not in item:
        if should_stop():return None
        saved['stage']='repairing';checkpoint(saved)
        base_seed=(spec['seed']+saved['attempt']*7919)%(2**31-1)
        seed=int(g.digest([base_seed,repair.VERSION])[:8],16)%(2**31-1)
        replacement,receipt=client.chat(repair.messages(original['authored'],spec,item['targets']),
            repair.schema(item['targets']),seed,role='author',max_tokens=1536)
        item.update(replacements=replacement,receipt=receipt,replacement_sha256=g.digest(replacement),receipt_sha256=g.digest(receipt))
        saved['stage']='repair_received';checkpoint(saved)
    hashes=[original['author_receipt']['request_sha256'],item['receipt']['request_sha256']]
    saved.update(authored=repair.apply(original['authored'],spec,item['targets'],item['replacements']),
        author_receipt={'type':'original_plus_one_title_repair','request_sha256':g.digest(hashes),
                        'original_request_sha256':hashes[0],'repair_request_sha256':hashes[1]},stage='repaired')
    verify(saved,complete=True);checkpoint(saved)
    episode=g.normalize(saved['authored'],spec,saved['author_receipt']);g.validate_corpus([episode])
    return episode
