import json
import os

senttag2opinion = {'pos': 'great', 'neg': 'bad', 'neu': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

rest_aspect_cate_list = [
    'location general', 'food prices', 'food quality', 'food general',
    'ambience general', 'service general', 'restaurant prices',
    'drinks prices', 'restaurant miscellaneous', 'drinks quality',
    'drinks style_options', 'restaurant general', 'food style_options'
]

laptop_aspect_cate_list = [
    'keyboard operation_performance', 'os operation_performance',
    'out_of_scope operation_performance', 'ports general',
    'optical_drives general', 'laptop operation_performance',
    'optical_drives operation_performance', 'optical_drives usability',
    'multimedia_devices general', 'keyboard general', 'os miscellaneous',
    'software operation_performance', 'display operation_performance',
    'shipping quality', 'hard_disc quality', 'motherboard general',
    'graphics general', 'multimedia_devices connectivity', 'display general',
    'memory operation_performance', 'os design_features',
    'out_of_scope usability', 'software design_features',
    'graphics design_features', 'ports connectivity',
    'support design_features', 'display quality', 'software price',
    'shipping general', 'graphics operation_performance',
    'hard_disc miscellaneous', 'display design_features',
    'cpu operation_performance', 'mouse general', 'keyboard portability',
    'hardware price', 'support quality', 'hardware quality',
    'motherboard operation_performance', 'multimedia_devices quality',
    'battery design_features', 'mouse usability', 'os price',
    'shipping operation_performance', 'laptop quality', 'laptop portability',
    'fans&cooling general', 'battery general', 'os usability',
    'hardware usability', 'optical_drives design_features',
    'fans&cooling operation_performance', 'memory general', 'company general',
    'power_supply general', 'hardware general', 'mouse design_features',
    'software general', 'keyboard quality', 'power_supply quality',
    'software quality', 'multimedia_devices usability',
    'power_supply connectivity', 'multimedia_devices price',
    'multimedia_devices operation_performance', 'ports design_features',
    'hardware operation_performance', 'shipping price',
    'hardware design_features', 'memory usability', 'cpu quality',
    'ports quality', 'ports portability', 'motherboard quality',
    'display price', 'os quality', 'graphics usability', 'cpu design_features',
    'hard_disc general', 'hard_disc operation_performance', 'battery quality',
    'laptop usability', 'company design_features',
    'company operation_performance', 'support general', 'fans&cooling quality',
    'memory design_features', 'ports usability', 'hard_disc design_features',
    'power_supply design_features', 'keyboard miscellaneous',
    'laptop miscellaneous', 'keyboard usability', 'cpu price',
    'laptop design_features', 'keyboard price', 'warranty quality',
    'display usability', 'support price', 'cpu general',
    'out_of_scope design_features', 'out_of_scope general',
    'software usability', 'laptop general', 'warranty general',
    'company price', 'ports operation_performance',
    'power_supply operation_performance', 'keyboard design_features',
    'support operation_performance', 'hard_disc usability', 'os general',
    'company quality', 'memory quality', 'software portability',
    'fans&cooling design_features', 'multimedia_devices design_features',
    'laptop connectivity', 'battery operation_performance', 'hard_disc price',
    'laptop price'
]

books_aspect_cate_list = ['book author', 'content plot', 'book quality', 'none', 'book audience', 'service general', 'book length', 'book structure', 'derivatives general', 'book general', 'content characters', 'content genre', 'book prices', 'book title']

clothing_aspect_cate_list = ['socks size', 'bottom brand', 'top prices', 'top quality', 'shoes size', 'bottom prices', 'bottom quality', 'top general', 'service size', 'bottom pair', 'bottom looking', 'socks quality', 'clothing size', 'top options', 'bottom size', 'bottom general', 'top pair', 'clothing general', 'clothing brand', 'clothing pair', 'shoes options', 'clothing prices', 'clothing quality', 'shoes prices', 'shoes looking', 'shoes pair', 'top size', 'service general', 'top looking', 'socks looking', 'clothing looking', 'socks general', 'shoes brand', 'shoes quality', 'shoes general']

hotel_aspect_cate_list = ['rooms general', 'hotel comfort', 'facilities cleanliness', 'room_amenities quality', 'location general', 'food_drinks general', 'room_amenities cleanliness', 'hotel general', 'rooms design_features', 'rooms cleanliness', 'facilities design_features', 'hotel cleanliness', 'food_drinks style_options', 'facilities prices', 'hotel design_features', 'hotel prices', 'rooms quality', 'room_amenities design_features', 'hotel quality', 'room_amenities comfort', 'rooms comfort', 'facilities comfort', 'facilities general', 'room_amenities general', 'service general', 'food_drinks quality', 'facilities quality']

with open("force_tokens.json", 'r') as f:
    force_tokens = json.load(f)

with open("force_tokens_full.json", 'r') as f:
        force_tokens_full = json.load(f)


cate_list = {
    "R15": rest_aspect_cate_list,
    "R16": rest_aspect_cate_list,
    "Lap": laptop_aspect_cate_list,
    "Rest": rest_aspect_cate_list,
    "M-Rest": rest_aspect_cate_list,
    "M-Lap": laptop_aspect_cate_list,
    "Books": books_aspect_cate_list,
    "Clothing": clothing_aspect_cate_list,
    "Hotel": hotel_aspect_cate_list
}

task_data_list = {
    "memd": ["Books", "Clothing", "Hotel"],
    "acos": ['Lap', "Rest"],
    "asqp": ['R15', "R16"],
}
force_words = {
    'memd': {
        "books": books_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "clothing": clothing_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "hotel": hotel_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
    },
    'acos': {
        "rest": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "laptop": laptop_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
    },
    'asqp': {
        "rest15": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
        "rest16": rest_aspect_cate_list + list(sentword2opinion.values()) + ['[SSEP]'],
    }
}