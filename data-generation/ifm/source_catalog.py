"""Research-oriented sampling policy. Weights are design choices, not traffic estimates."""
VERSION = 'research_discovery_sources_v1'
RESEARCHED_AT = '2026-09-09'
# session share, relevant source pool, author objective
NICHES = {
    'market_and_business': (20, ('search', 'reviews', 'product_discovery', 'company_intelligence', 'linkedin', 'company', 'article', 'youtube', 'community', 'independent'),
        'investigate customer problems, competing offers, market positioning and evidence of demand'),
    'products_and_suppliers': (15, ('search', 'amazon', 'alibaba', 'marketplaces', 'company', 'youtube', 'community', 'article', 'independent'),
        'research product features, supplier capabilities, procurement requirements and tradeoffs'),
    'people_and_freelancers': (15, ('search', 'linkedin', 'portfolio', 'company_intelligence', 'company', 'article', 'youtube', 'independent'),
        'evaluate fictional experts, founders, freelancers or agencies using relevant public professional work'),
    'creators_and_social': (15, ('search', 'youtube', 'instagram', 'x', 'tiktok', 'facebook', 'twitch', 'community', 'article', 'independent'),
        'discover fictional creators, audience interests, content angles and potential collaborations'),
    'papers_and_evidence': (15, ('search', 'scholarly', 'papers', 'archives', 'encyclopedia', 'article', 'youtube', 'independent'),
        'investigate a precise question using research, reports, explanatory sources and conflicting claims'),
    'technical_research': (10, ('search', 'github', 'technical', 'hackernews', 'research', 'scholarly', 'article', 'youtube', 'independent'),
        'research a technical tool, integration or method supporting this domain, including limitations and alternatives'),
    'design_and_portfolios': (5, ('search', 'design', 'pinterest', 'instagram', 'portfolio', 'article', 'company', 'independent'),
        'discover visual references and compare fictional design portfolios for a concrete project brief'),
    'open_web_discovery': (5, ('search', 'encyclopedia', 'archives', 'article', 'medium', 'youtube', 'community', 'independent'),
        'explore an unfamiliar subject through complementary references and independent accounts'),
}
STAGES = {
    'discovery': (35, 'Find promising sources, entities or directions for a concrete question.'),
    'comparison': (30, 'Compare alternatives using different criteria and evidence.'),
    'evidence_checking': (25, 'Check an assertion by gathering corroborating and conflicting evidence; do not group by agreement.'),
    'project_followup': (10, 'Organize references needed to act on an existing project or plan.'),
}


def select(mapping, bucket):
    assert 0 <= bucket < 100 and sum(value[0] for value in mapping.values()) == 100
    for name, value in mapping.items():
        bucket -= value[0]
        if bucket < 0:
            return name
    raise AssertionError('unreachable sampling bucket')


def policy_summary():
    return {'version': VERSION, 'research_date_utc': RESEARCHED_AT,
        'weights_are': 'planned_session_percent_not_measured_user_traffic',
        'niches': {key: value[0] for key, value in NICHES.items()},
        'stages': {key: value[0] for key, value in STAGES.items()}}
