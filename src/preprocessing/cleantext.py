import re


def clean_tweet(tweet_text: str,
                remove_users: bool = False,
                remove_hashtags: bool = False) -> str:
    # lower case
    cleaned = tweet_text.lower()

    # remove non utf-8
    cleaned = bytes(cleaned, 'utf-8').decode('utf-8', 'ignore')

    # replace URLs by a single word that represents URL
    cleaned = re.sub(r'((http://www\.|https://www\.|http://|https://)?' +
                     r'[a-z0-9]+([\-.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(/.*)?)', r'url', cleaned)

    # remove user tags, like @User1
    if remove_users:
        cleaned = re.sub(r'(@[\w_-]+)', r' ', cleaned)

    # Remove hashtags, like #hashtash
    if remove_hashtags:
        cleaned = re.sub(r'(#[\w_-]+)', r' ', cleaned)

    # remove bad characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', r' ', cleaned)

    # remove double spaces
    cleaned = re.sub(r' +', r' ', cleaned)

    return cleaned.strip()
