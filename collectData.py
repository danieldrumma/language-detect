
import wikipedia
import re
import string
"""
Extract data from wikipedia
"""

def grab_article_en():
    articles = wikipedia.random(pages=10)
    return articles


def grab_article_nl():
    wikipedia.set_lang(prefix="nl")
    articles = wikipedia.random(pages=10)
    return articles


def divide_strings(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_article(page_title):
    try:
        page = wikipedia.page(page_title)
    except wikipedia.exceptions.DisambiguationError:
        return None
    except  wikipedia.exceptions.PageError:
        return None

    try:
        content = page.content
        content = re.sub("== [\w\s]+ ==", '', content)
        text = content.strip('\n')
        text = text.replace('\n','').split(' ')
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in text]
        text = list(divide_strings(stripped, 15))
        del text[-1]
        return text
    except wikipedia.exceptions.PageError:
        print("Page not found")
        return None


def get_my_articles_en():
    en_articles = grab_article_en()
    content = []
    print("New set of English")
    for i in en_articles:
        data = process_article(i)
        if data is not None:
            content.extend(data)
    return content


def get_my_articles_nl():
    en_articles = grab_article_nl()
    content = []
    print("New set of dutch")
    for i in en_articles:
        data = process_article(i)
        if data is not None:
            content.extend(data)
    return content


def write_to_files(test, train, data, lang):
    for item in data:
        word = ""
        for i in item:
            word = word + i + " "
        word = word.strip()
        test.write(word + '\n')
        if lang == "en":
            word = "en|" + word + '\n'
            train.write(word)
        else:
            word = "nl|" + word + '\n'
            train.write(word)


def main():
    data_main = []
    train = open('train.dat', 'w',  encoding="utf-8")
    test = open('test.dat', 'w', encoding="utf-8")
    for i in range(0, 15):
        if i % 5 == 0:
            print("Computing....")
        en_data = get_my_articles_en()
        write_to_files(test, train, en_data, "en")
        nl_data = get_my_articles_nl()
        write_to_files(test, train, nl_data, "nl")
    test.close()
    train.close()


if __name__ == "__main__":
    main()