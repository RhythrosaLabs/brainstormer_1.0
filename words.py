import random


# A list of sample words to generate random sentences
sample_words = [
    
    # Adjectives and Descriptors
    'glistening', 'metallic', 'highly detailed', 'stunning', 'beautiful', 'cosmic',
    'geometric', 'glorious', 'magical', 'hyperrealistic', 'cinematic', 'celestial',
    'robotic', 'ethereal', 'majestic', 'whimsical', 'futuristic', 'vibrant', 'dreamlike',
    'nocturnal', 'lush', 'intricate', 'digital art', 'photorealistic', 'dancing', '3d render',
    'doodle', 'explorer', 'wandering aimlessly', 'retrofuturistic', 'sci-fi', 'abstract art',
    'post-apocalyptic', 'haunted', 'mirrored', 'bioluminescent', 'mysterious', 'surreal', 'vivid',

# Colors
    'magenta', 'orange', 'teal', 'turquoise', 'crimson', 'emerald', 'diamond', 'golden',
    'silver', 'rose golden', 'yellow', 'blue and orange', 'yellow and orange', 'teal and orange',
    'neon',

# Animals and Plants
'bird', 'fish', 'lion', 'tiger', 'elephant', 'giraffe', 'dinosaur', 'whale', 'eagle', 'leviathan',
'phoenix', 'toucan', 'peacock', 'chameleon', 'gargoyle', 'cherry blossom', 'bonsai', 'lotus',
'lily pad', 'mushroom', 'water lily' 'panda', 'cheetah', 'rhinoceros', 'gorilla', 'kangaroo', 'chimpanzee', 'koala',
'zebra', 'hippopotamus', 'platypus', 'octopus', 'jellyfish', 'dolphin',
'starfish', 'seahorse', 'penguin', 'flamingo', 'wolf', 'crocodile', 'alligator',
'fox', 'gazelle', 'gibbon', 'hyena', 'jaguar', 'leopard', 'lynx', 'manta ray',
'orca', 'osprey', 'otter', 'owl', 'parrot', 'pelican', 'polar bear', 'porcupine',
'puma', 'raccoon', 'raven', 'reindeer', 'salamander', 'seagull', 'shark', 'sloth',  'whale', 'eagle', 
'snake', 'snow leopard', 'sparrow', 'squid', 'squirrel', 'stingray', 'swan',
'tapir', 'tarpon', 'toucan', 'turtle', 'vulture', 'walrus', 'wolverine', 'hummingbird',
'acacia', 'aloe', 'azalea', 'bamboo', 'begonia', 'bougainvillea', 'cactus',
'camellia', 'carnation', 'cedar', 'chrysanthemum', 'clematis', 'clover', 'crocus',
'daffodil', 'dahlia', 'daisy', 'edelweiss', 'fern', 'fir', 'fuchsia', 'gardenia',
'geranium', 'gingko', 'gladiolus', 'hawthorn', 'heather', 'hibiscus', 'holly',
'honeysuckle', 'hosta', 'hydrangea', 'iris', 'ivy', 'jasmine', 'juniper', 'lantana',
'lavender', 'lilac', 'lily', 'magnolia', 'marigold', 'mimosa', 'mint', 'moss',
'narcissus', 'oak', 'orchid', 'pansy', 'peony', 'periwinkle', 'petunia', 'pine',
'poppy', 'rhododendron', 'rose', 'rosemary', 'sage', 'snapdragon', 'sunflower',
'sweet pea', 'tulip', 'verbena', 'violet', 'wisteria', 'yucca', 'zinnia', 'toucan', 'peacock', 
'apple', 'peaches', 'cherry blossom', 'bonsai', 

    

# Places
    'river', 'mountain', 'forest', 'desert', 'orchard', 'volcano', 'iceberg', 'pagoda',
    'maze', 'cathedral', 'dragon den', 'statue', 'monolith', 'butterfly', 'gondola', 'gazebo',
    'clock tower', 'airship', 'hot air balloon', 'coral reef', 'hedge maze', 'treasure cove', 'dreamland',
    'abandoned factory', 'temple', 'aquarium', 'watermill', 'flower field', 'cityscape',
    'bamboo forest', 'greenhouse', 'marketplace', 'library', 'wooden bridge', 'harbor', 'theater', 'opera house',
    'art studio', 'bookstore', 'parade', 'zen garden', 'amusement park', 'farm', 'vineyard', 'fountain', 'koi pond',
    'windmill', 'snowscape', 'beach', 'campfire', 'ferris wheel', 'constellation', 'mansion',
    'carriage', 'ancient temple', 'sphinx', 'golden gate', 'rose garden', 'colosseum', 'submarine', 'sailboat',
    'sculpture museum', 'treehouse', 'crystal cave', 'carousel', 'topiary', 'pirate ship', 'coral reef', 'pacific rim',
    'cavern', 'dome', 'pavilion', 'lab', 'cabin', 'fortress', 'ruins', 'obelisk', 'monastery', 'mausoleum',
    'aqueduct', 'labyrinth', 'terrarium', 'arcade','menagerie', 'gate', 'archway',
    'koi pond', 'pier', 'pylon', 'gate', 'archway', 'monastery', 'mausoleum', 'lab', 'cabin', 'fortress',
    'rose garden', 'carousel', 'topiary', 'pirate ship', 'reef', 'cavern', 'dome', 'pavilion', 'wishing well',
    'meadow', 'cathedral', 'colosseum', 'observatory', 'mysterious door', 'horizon',
 

#Weather
'wind', 'rain', 'thunder', 'lightning', 'hurricane', 'tornado'


#Prepositions and actions
'in a valley', 'with plumes of smoke', 'at sunrise', 'at sunset', 'in las vegas', 'in a museum',
'in a mansion', 'inside of a bubble', 'inside of a factory', 'on the moon', 'on planet saturn',
'in outer space', 'inside an enchanted forest', 'while swimming underwater', 'on a floating island', 'inside of hidden cave',
'in the clouds', 'in a forgotten city', 'in a dream', 'in a library', 'in a spaceship',
'in a prehistoric era', 'climbing a staircase', 'holding a balloon', 'inside of an igloo',
    'staring into the vast beyond', 'making music', 'playing an instrument', 'performing science experiments',
    'performing magic tricks', 'engineering', 'tinkering', 'floating', 'gushing', 'soaring', 'flying', 'resting',
    'in a spaceship', 'in self-expression', 'calculating', 'equating', 'determining', 'fixing', 'tweaking',
    'creating', 'making', 'designing', 'building', 'crafting', 'doing', 'thinking', 'pondering', 'wondering',
    'pondering', 'meandering', 'shuffling', 'smirking', 'in awe', 'in love', 'in bliss', 'beyond comprehension',
    'the meaning of life', 'the key to life', 'answering', 'searching', 'looking', 'discovering',


#Celestial
'sun', 'above the stars', 'galaxy', 'crescent moon', 'nebula', 'vortex', 'portal', 'comet', 'meteor', 'asteroid', 'satellite', 'rocket', 
'starry sky', 'moonlit night', 'sunrise', 'sunset', 'ocean waves', 'desert sands', 'snow-capped mountains',
'rainforest', 'waterfall', 'river', 'lake', 'canyon', 'volcano', 'geyser', 'hot spring', 'lava flow',
'thunderstorm', 'lightning bolt', 'rainbow', 'northern lights', 'constellation', 'comet', 'meteor shower',
'nebula', 'galaxy', 'black hole', 'supernova', 'space station', 'rocket', 'alien', 'extraterrestrial',
'spaceship', 'time travel', 'portal', 'dimension', 'parallel universe', 'quantum', 'wormhole', 'teleportation',
'hologram', 'virtual reality', 'cyberspace', 'artificial intelligence', 'robot', 'android', 'cyborg',
'nanotechnology', 'replicant', 'utopia', 'dystopia', 'clone', 'genetic engineering', 'mutation', 'biotechnology',
'apocalypse', 'posthuman', 'transhuman', 'singularity', 'simulation', 'augmented reality', 'mind uploading',
'cyberpunk', 'steampunk', 'biopunk', 'solarpunk', 'lunar colony', 'martian colony', 'interstellar', 'terraforming',
'dark matter', 'antimatter', 'multiverse', 'zero gravity', 'holographic', 'subatomic', 'supercomputer',
'techno-organic', 'hyperspace', 'cryogenic', 'synthetic', 'bionic', 'telepathic', 'psionic', 'telekinetic',
'psychokinetic', 'cybernetic', 'sentient', 'quantum entanglement', 'holodeck', 'replicator', 'force field',
'cloaking device', 'warp drive', 'tractor beam', 'phaser', 'teleporter', 'deflector shield', 'energy beam',
'stasis chamber', 'exo-suit', 'energy sword', 'plasma cannon', 'laser blaster', 'neutron star', 'pulsar',
'quasar', 'white dwarf', 'red giant', 'black dwarf', 'blue supergiant', 'brown dwarf', 'red dwarf',
'protostar', 'supergiant', 'hypergiant', 'variable star', 'binary star', 'trinary star', 'neutron star',
'main sequence', 'stellar nursery', 'star cluster', 'globular cluster', 'open cluster', 'intergalactic',
'star system', 'solar system', 'planetary system', 'asteroid belt', 'kuiper belt', 'oort cloud', 'cometary',
'meteoroid', 'meteorite',


#Stuff
'hourglass','obelisk', 'unicycle',  'lantern', 'henge', 'seashell', 'coral',
'aqueduct','carriage','grand piano', 'sculpture', 'pylon', 'gargoyle', 'fresco', 'pinnacle',

#Gods
'Chronos', 'Ra', 'Tyche', 'Hecate', 'Druidic Deity', 'Aphrodite', 'Poseidon',
'Nereus', 'Hermes', 'Apollo', 'Euterpe', 'Hephaestus', 'Zeus',
'Daedalus', 'Leviathan', 'Phoenix', 'Dionysus', 'Gargoyle Deity', 'Fresco Deity',
'Pinnacle Deity', 'Artemis', 'Athena', 'Hades', 'Aeolus', 'Nyx', 'Persephone',
'Demeter', 'Hera', 'Ares', 'Eros', 'Pan', 'Bacchus', 'Jupiter',
'Janus', 'Mars', 'Venus', 'Mercury', 'Neptune', 'Pluto', 'Vesta',
'Cupid', 'Faunus', 'Fortuna', 'Minerva', 'Vulcan', 'Ceres', 'Diana',
'Juno', 'Saturn', 'Uranus', 'Anubis', 'Osiris', 'Isis', 'Horus',
'Seth', 'Thoth', 'Bastet', 'Sekhmet', 'Amun', 'Ptah', 'Khnum',
'Hathor', 'Maat', 'Nephthys', 'Sobek', 'Mut', 'Geb', 'Nut',
'Tefnut', 'Shu', 'Atum', 'Khepri', 'Serapis', 'Apis', 'Neith'
    
]

#----------------------------------------------------------------------------------------------------------------------

# RANDOMIZER 

def generate_random_sentence():
    #generates a random sentence based on the words list
    sentence_length = random.randint(6, 12)
    sentence = ' '.join(random.choice(sample_words) for _ in range(sentence_length))
    return sentence

#----------------------------------------------------------------------------------------------------------------------



