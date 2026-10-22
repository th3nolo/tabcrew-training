"""Hand-designed theme catalog (pairs of unrelated topics per session) used by generate.py.

Splits are fixed: train, validation and final_test themes never overlap.
"""
FAMILIES = {
 'train': [
  'Backend reliability and urban balcony gardening', 'Database performance and jazz rehearsal',
  'Web accessibility and trail running', 'Queue processing and home bread baking',
  'Container deployment and aquarium care', 'Network troubleshooting and sewing clothes',
  'Mobile application testing and ceramic glazing', 'Data visualization and cycling commutes',
  'API design and coffee preparation', 'Game development and dog training',
  'Embedded electronics and vegetable fermentation', 'Photo editing and camping equipment',
  'Video production and indoor plant care', 'Audio recording and woodworking joinery',
  'Graphic illustration and swimming practice', 'Animation workflows and telescope observation',
  'Typography projects and cooking for guests', 'Digital painting and climbing equipment',
  'Music composition and bicycle repair', 'Podcast editing and house painting',
  'Apartment organization and language learning', 'Household appliance repair and kayaking trips',
  'Interior lighting design and chess practice', 'Kitchen renovation and birdwatching',
  'Rainwater collection and local history research', 'Solar energy learning and origami projects',
  'Science fair preparation and dance rehearsals', 'University statistics and backpacking routes',
  'University literature and planted terrariums', 'School mathematics and tabletop gaming',
  'Geography study and homemade cheese', 'Chemistry coursework and fountain pen maintenance',
  'Conference logistics and textile dyeing', 'Community volunteering and sourdough baking',
  'Wedding stationery and landscape photography', 'Neighborhood newsletter and weight training',
  'Train travel planning and miniature painting', 'Museum trip planning and knitting patterns',
  'Language exchange events and weather observation', 'Theater rehearsal and family recipe archiving'],
 'validation': ['Beekeeping and badminton practice', 'Printmaking and sailing lessons',
  'Choir rehearsal and orchid care', 'Weaving on a loom and squash practice',
  'Archival genealogy and juggling routines'],
 'final_test': ['Book restoration and roller skating', 'Glass mosaics and table tennis',
  'Leather craft and kite flying', 'Piano regulation and tea tasting',
  'Bonsai shaping and model railway construction']}
