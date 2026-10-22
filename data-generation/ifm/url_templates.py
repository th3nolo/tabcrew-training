"""Offline synthetic URLs with recognizable platform routes; never fetch these URLs."""
import base64
from collections import Counter
import hashlib
import re
import unicodedata
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

VERSION = 'platform_routes_v1'
# platform, hostname, route format, author-facing page description, title suffix
ROUTES = {
    'google_search': ('google', 'www.google.com', '/search?q={query}', 'Google search results for a specific research question', 'Google Search'),
    'bing_search': ('bing', 'www.bing.com', '/search?q={query}', 'Bing search results for a specific research question', 'Bing'),
    'duckduckgo_search': ('duckduckgo', 'duckduckgo.com', '/?q={query}', 'DuckDuckGo search results for a specific research question', 'DuckDuckGo'),
    'scholar_search': ('google_scholar', 'scholar.google.com', '/scholar?q={query}', 'Google Scholar search results for a precise evidence question', 'Google Scholar'),
    'semantic_paper': ('semantic_scholar', 'www.semanticscholar.org', '/paper/{slug}/{hex40}', 'a scientific paper record: a specific research question or finding', 'Semantic Scholar'),
    'researchgate_paper': ('researchgate', 'www.researchgate.net', '/publication/{publication}_{underscored}', 'a research publication record with a descriptive study title', 'ResearchGate'),
    'archive_item': ('internet_archive', 'archive.org', '/details/{slug}', 'an archived report, book or historical reference with a descriptive title', 'Internet Archive'),
    'wikipedia_en': ('wikipedia', 'en.wikipedia.org', '/wiki/{underscored}', 'an English encyclopedia entry: name a specific relevant concept, not a made-up study', 'Wikipedia'),
    'wikipedia_es': ('wikipedia', 'es.wikipedia.org', '/wiki/{underscored}', 'a Spanish encyclopedia entry: name a specific relevant concept, not a made-up study', 'Wikipedia'),
    'g2_reviews': ('g2', 'www.g2.com', '/products/{slug}/reviews', 'a review page comparing features and limitations of fictional business software', 'G2'),
    'producthunt_product': ('product_hunt', 'www.producthunt.com', '/products/{slug}', 'a discovery page for a fictional product and its concrete purpose', 'Product Hunt'),
    'crunchbase_company': ('crunchbase', 'www.crunchbase.com', '/organization/{person}', 'a fictional company record: name followed by industry and business purpose', 'Crunchbase'),
    'behance_project': ('behance', 'www.behance.net', '/gallery/{publication}/{slug}', 'a fictional design portfolio project with a specific visual deliverable', 'Behance'),
    'dribbble_shot': ('dribbble', 'dribbble.com', '/shots/{question}-{slug}', 'a fictional design portfolio shot with a specific design purpose', 'Dribbble'),
    'pinterest_pin': ('pinterest', 'www.pinterest.com', '/pin/{pin}/', 'a saved visual reference for a specific design, product or project', 'Pinterest'),
    'tiktok_video': ('tiktok', 'www.tiktok.com', '/@{handle}/video/{tiktok}', 'a short video: concrete creator observation, demonstration or comparison', 'TikTok'),
    'tiktok_profile': ('tiktok', 'www.tiktok.com', '/@{handle}', 'a fictional creator profile: name followed by a relevant content specialty', 'TikTok'),
    'facebook_page': ('facebook', 'www.facebook.com', '/{handle}/', 'a fictional public professional page: name followed by business or creator specialty', 'Facebook'),
    'mercadolibre_product': ('mercado_libre', 'www.mercadolibre.com.ar', '/{slug}/p/MLA{question}', 'a Mercado Libre product listing: fictional product name and concrete specifications', 'Mercado Libre'),
    'ebay_product': ('ebay', 'www.ebay.com', '/itm/{listing}', 'an eBay listing: fictional product or equipment with concrete features', 'eBay'),
    'etsy_product': ('etsy', 'www.etsy.com', '/listing/{vod}/{slug}', 'an Etsy listing: a fictional handmade product or design with concrete features', 'Etsy'),
    'hackernews_item': ('hacker_news', 'news.ycombinator.com', '/item?id={question}', 'a Hacker News discussion: a specific technical observation, project or question', 'Hacker News'),
    'youtube_video': ('youtube', 'www.youtube.com', '/watch?v={video}', 'a YouTube video: a natural interview, review or tutorial title', 'YouTube'),
    'youtube_short': ('youtube', 'www.youtube.com', '/shorts/{video}', 'a YouTube Short: a concise demo or observation', 'YouTube'),
    'youtube_channel': ('youtube', 'www.youtube.com', '/@{handle}', 'a YouTube channel: fictional creator name followed by a relevant specialty', 'YouTube'),
    'youtube_search': ('youtube', 'www.youtube.com', '/results?search_query={query}', 'YouTube search results for a concrete research question', 'YouTube'),
    'linkedin_profile': ('linkedin', 'www.linkedin.com', '/in/{person}/', 'a LinkedIn profile: fictional full name — relevant professional role', 'LinkedIn'),
    'linkedin_company': ('linkedin', 'www.linkedin.com', '/company/{person}/', 'a LinkedIn company profile: fictional business name — concrete specialty', 'LinkedIn'),
    'linkedin_post': ('linkedin', 'www.linkedin.com', '/posts/{person}_{slug}-activity-{post}-aB7c', 'a LinkedIn post with a specific professional observation or case study', 'LinkedIn'),
    'x_post': ('x', 'x.com', '/{handle}/status/{post}', 'an X post: a concise public professional observation or question', 'X'),
    'twitter_post': ('x', 'twitter.com', '/{handle}/status/{post}', 'a Twitter post: a concise public professional observation or question', 'X'),
    'x_profile': ('x', 'x.com', '/{handle}', 'an X profile: fictional name — professional interests relevant to the objective', 'X'),
    'x_search': ('x', 'x.com', '/search?q={query}&src=typed_query', 'X search results for a concrete topic, person specialty or product', 'X'),
    'medium_story': ('medium', 'medium.com', '/@{person}/{slug}-{hex12}', 'a Medium story with a natural descriptive article headline', 'Medium'),
    'medium_profile': ('medium', 'medium.com', '/@{person}', 'a Medium author profile: fictional name — field of writing', 'Medium'),
    'medium_publication': ('medium', 'medium.com', '/{person}/{slug}-{hex12}', 'an article in a fictional Medium publication', 'Medium'),
    'amazon_product': ('amazon', 'www.amazon.com', '/{slug}/dp/{asin}', 'an Amazon product listing: fictional product name and concrete features', 'Amazon.com'),
    'amazon_short_product': ('amazon', 'www.amazon.com', '/dp/{asin}', 'an Amazon product listing: fictional product name and useful specifications', 'Amazon.com'),
    'amazon_search': ('amazon', 'www.amazon.com', '/s?k={query}', 'Amazon search results for products relevant to the research objective', 'Amazon.com'),
    'alibaba_product': ('alibaba', 'www.alibaba.com', '/product-detail/{slug}_{product}.html', 'an Alibaba wholesale listing: product material, purpose and sourcing features', 'Alibaba.com'),
    'alibaba_search': ('alibaba', 'www.alibaba.com', '/trade/search?SearchText={query}', 'Alibaba search results for a concrete procurement requirement', 'Alibaba.com'),
    'twitch_channel': ('twitch', 'www.twitch.tv', '/{handle}', 'a Twitch channel: fictional creator name — stream specialty', 'Twitch'),
    'twitch_video': ('twitch', 'www.twitch.tv', '/videos/{vod}', 'a Twitch past broadcast: interview, workshop, demonstration or discussion', 'Twitch'),
    'twitch_clip': ('twitch', 'clips.twitch.tv', '/{clip}', 'a Twitch clip: one concrete moment from a demonstration or discussion', 'Twitch'),
    'instagram_profile': ('instagram', 'www.instagram.com', '/{handle}/', 'an Instagram professional profile: fictional name — relevant specialty', 'Instagram'),
    'instagram_post': ('instagram', 'www.instagram.com', '/p/{video}/', 'an Instagram post: a concrete product, project or professional observation', 'Instagram'),
    'instagram_reel': ('instagram', 'www.instagram.com', '/reel/{video}/', 'an Instagram reel: a concise demonstration, interview or comparison', 'Instagram'),
    'reddit_post': ('reddit', 'www.reddit.com', '/r/{community}/comments/{short}/{slug}/', 'a Reddit discussion: a natural question or account of an experience', 'Reddit'),
    'reddit_search': ('reddit', 'www.reddit.com', '/search/?q={query}', 'Reddit search results for a concrete problem or comparison', 'Reddit'),
    'github_repo': ('github', 'github.com', '/{handle}/{slug}', 'a GitHub repository: a fictional software project and its purpose', 'GitHub'),
    'github_issue': ('github', 'github.com', '/{handle}/{slug}/issues/{issue}', 'a GitHub issue: a specific software integration problem or feature request', 'GitHub'),
    'substack_story': ('substack', '{person}.substack.com', '/p/{slug}', 'a Substack newsletter article with a concrete research insight', 'Substack'),
    'upwork_profile': ('upwork', 'www.upwork.com', '/freelancers/~{freelancer}', 'an Upwork freelancer profile: fictional full name — service specialty', 'Upwork'),
    'fiverr_gig': ('fiverr', 'www.fiverr.com', '/{handle}/{slug}', 'a Fiverr service listing: a specific freelance deliverable', 'Fiverr'),
    'arxiv_abstract': ('arxiv', 'arxiv.org', '/abs/{paper}', 'a fictional research-paper abstract with a specific research question', 'arXiv'),
    'arxiv_pdf': ('arxiv', 'arxiv.org', '/pdf/{paper}', 'a fictional research-paper PDF with a specific finding or method', 'arXiv'),
    'stackoverflow_question': ('stackoverflow', 'stackoverflow.com', '/questions/{question}/{slug}', 'a Stack Overflow question about a concrete software problem', 'Stack Overflow'),
    'company_product': ('independent_site', 'harborworks.example', '/products/{slug}', 'an independent company product page with useful specifications', ''),
    'company_case': ('independent_site', 'harborworks.example', '/case-studies/{slug}', 'a company case study with a concrete outcome or constraint', ''),
    'independent_article': ('independent_site', 'fieldnotes.example', '/articles/{slug}', 'an article on an independent website with a descriptive headline', ''),
    'portfolio_project': ('independent_site', 'northstarstudio.example', '/work/{slug}', 'a freelancer portfolio project with a concrete deliverable', ''),
    'independent_pdf': ('independent_site', 'fieldnotes.example', '/resources/{slug}.pdf', 'a PDF industry report with a concrete research question', ''),
}
OPTIONS = {
    'search': ('google_search', 'bing_search', 'duckduckgo_search'),
    'scholarly': ('scholar_search',),
    'papers': ('semantic_paper', 'researchgate_paper', 'independent_pdf'),
    'archives': ('archive_item',),
    'encyclopedia': ('wikipedia_en', 'wikipedia_es'),
    'reviews': ('g2_reviews',),
    'product_discovery': ('producthunt_product',),
    'company_intelligence': ('crunchbase_company',),
    'design': ('behance_project', 'dribbble_shot'),
    'pinterest': ('pinterest_pin',),
    'tiktok': ('tiktok_video', 'tiktok_profile'),
    'facebook': ('facebook_page',),
    'marketplaces': ('mercadolibre_product', 'ebay_product', 'etsy_product'),
    'hackernews': ('hackernews_item',),
    'independent': ('company_product', 'company_case', 'independent_article', 'portfolio_project', 'independent_pdf'),
    'youtube': ('youtube_video', 'youtube_short', 'youtube_channel', 'youtube_search'),
    'linkedin': ('linkedin_profile', 'linkedin_company', 'linkedin_post'),
    'x': ('x_post', 'twitter_post', 'x_profile', 'x_search'),
    'medium': ('medium_story', 'medium_profile', 'medium_publication'),
    'amazon': ('amazon_product', 'amazon_short_product', 'amazon_search'),
    'alibaba': ('alibaba_product', 'alibaba_search'),
    'twitch': ('twitch_channel', 'twitch_video', 'twitch_clip'),
    'instagram': ('instagram_profile', 'instagram_post', 'instagram_reel'),
    'community': ('reddit_post', 'reddit_search'),
    'github': ('github_repo', 'github_issue'),
    'article': ('medium_story', 'substack_story', 'independent_article'),
    'research': ('arxiv_abstract', 'arxiv_pdf', 'independent_pdf'),
    'company': ('company_product', 'company_case'),
    'portfolio': ('portfolio_project', 'upwork_profile', 'fiverr_gig'),
    'technical': ('stackoverflow_question',),
}
TRACKING_KEYS = frozenset(('utm_source', 'utm_medium', 'utm_campaign', 'ref', 'feature', 'si'))
MEANINGFUL_KEYS = frozenset(('v', 'search_query', 'q', 'k', 'SearchText', 'src', 't', 'id'))


def slug(text, maximum=68):
    value = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    value = re.sub(r'[^a-z0-9]+', '-', value).strip('-')[:maximum].rstrip('-')
    return value or 'overview'


def make_url(kind, title, seed, ref):
    """Only visible title plus neutral randomness; no split/family/group labels enter a URL."""
    platform, host, route, _, suffix = ROUTES[kind]
    digest = hashlib.sha256(f'{VERSION}/{seed}/{ref}/{title}'.encode()).digest()
    hexed = digest.hex()
    integer = int.from_bytes(digest[:8], 'big')
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip('=')
    entity = re.split(r'\s[—–|]\s', title, maxsplit=1)[0]
    person = slug(entity, 28) + '-' + hexed[:6]
    handle = slug(entity, 7).replace('-', '_') + '_' + hexed[:6]
    fields = dict(slug=slug(title), person=person, handle=handle,
        underscored=quote('_'.join(title.split()), safe='_'), hex40=hexed[:40],
        publication=str(300000000 + integer % 99999999), listing=str(100000000000 + integer % 899999999999),
        pin=str(600000000000000000 + integer % 99999999999999999),
        tiktok=str(7000000000000000000 + integer % 99999999999999999),
        query=quote(' '.join(title.split()[:10]), safe=''), hex12=hexed[:12], video=encoded[:11],
        post=str(1800000000000000000 + integer % 99999999999999999),
        asin='B0' + hexed[:8].upper(), product=str(1600000000000 + integer % 99999999999),
        vod=str(1000000000 + integer % 8999999999), clip=slug(title, 32).replace('-', '') + encoded[:12],
        community='indiebusiness' if integer % 2 else 'research', short=hexed[:7],
        freelancer=hexed[:18],
        issue=str(integer % 499 + 1), question=str(10000000 + integer % 89999999))
    # arXiv YYMM dates must have a month in 01..12.
    fields['paper'] = f'{23 + integer % 3:02d}{1 + integer % 12:02d}.{integer % 99999:05d}'
    result = 'https://' + host.format(**fields) + route.format(**fields)
    if integer % 5 == 0:
        result += ('&' if '?' in result else '?') + 'utm_source=share&utm_medium=web'
    if kind == 'youtube_video' and integer % 4 == 0:
        result += '&t=' + str(integer % 600)
    if kind in ('independent_article', 'company_case') and integer % 4 == 0:
        result += '#overview'
    if kind == 'arxiv_pdf' and integer % 3 == 0:
        result += '#page=' + str(integer % 12 + 1)
    return result, (suffix if integer % 4 else '')


def canonical_page(url):
    parsed = urlsplit(url)
    host = (parsed.hostname or '').lower().removeprefix('www.')
    if host == 'twitter.com':
        host = 'x.com'
    pairs = [(k, v) for k, v in parse_qsl(parsed.query) if k not in TRACKING_KEYS and k not in ('t', 'src')]
    path = parsed.path.rstrip('/') or '/'
    if host == 'youtube.com' and path.startswith('/shorts/'):
        pairs = [('v', path.split('/')[2])]
        path = '/watch'
    if host == 'amazon.com' and '/dp/' in path:
        path = '/dp/' + path.split('/dp/', 1)[1].split('/')[0]
    if host == 'arxiv.org' and path.startswith('/pdf/'):
        path = '/abs/' + path.split('/pdf/', 1)[1].removesuffix('.pdf')
    return urlunsplit(('https', host, path, urlencode(sorted(pairs)), ''))


def validate_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port:
        raise ValueError('unsafe_url_structure')
    if len(url) > 1200 or re.search(r'[\s\x00-\x1f\x7f]', url) or '\\' in url:
        raise ValueError('unsafe_url_characters')
    if not re.fullmatch(r'[a-z0-9.-]+\.[a-z]{2,}', parsed.hostname):
        raise ValueError('invalid_hostname')
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if any(k not in TRACKING_KEYS | MEANINGFUL_KEYS for k, _ in pairs):
        raise ValueError('unexpected_query_parameter')
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError('duplicate_query_parameter')
    if re.search(r'qwen-research-(?:train|validation|final[_-]test)', url, re.I):
        raise ValueError('hidden_split_marker_in_url')
    return parsed


def coverage(episodes):
    cohorts = Counter()
    platforms = Counter()
    routes = Counter()
    niches = Counter()
    stages = Counter()
    for episode in episodes:
        realistic = episode['provenance'].get('synthetic_url_generator') == VERSION
        cohorts['realistic_urls' if realistic else 'legacy_opaque_urls'] += 1
        if realistic:
            routes.update(episode['research_context']['url_routes'])
            platforms.update(episode['research_context']['url_platforms'])
            niches.update([episode['research_context'].get('niche', 'unknown')])
            stages.update([episode['research_context'].get('research_stage', 'unknown')])
    return {'cohorts': dict(cohorts), 'platforms': dict(platforms), 'routes': dict(routes),
            'niches': dict(niches), 'research_stages': dict(stages)}
