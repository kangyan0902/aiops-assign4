import json
from pre_processing.pre_processing import PreProessor

my_pre_precessor = PrePcoessor(max_length_tweet=40, max_length_dictionary=10000)

def lambda_handler(event, context):
    
    tweet = event["tweet"]
    
    features = my_pre_precessor.pre_process_text(tweet)
    
    print(features)
    
    
    # TODO implement
    return {
        'features': features
    }
