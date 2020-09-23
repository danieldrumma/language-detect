import itertools


class trainText:
    create_id = itertools.count()
    attributes = None

    def __init__(self, lang, text, weight):
        self.lang = lang
        self.text = text
        self.weight = weight
        self.id = next(self.create_id)
        self.attributes = {"zijn_presence": False, "niet_presence": False, "en_presence": False,
                           "the_presence": False, "ee_presence": False, "aa_presence": False,
                           "de_presence": False, "engl_length": False, "nl_length": False,
                           "het_presence": False, "vowel_comp": False}

    def __str__(self):
        print("ID: " + str(self.id) + " Language: " + self.lang + " Text: " + self.text + "Attributes")
        attribute_string = ""
        for i in self.attributes:
            attribute_string += i + ": " + str(self.attributes[i]) + "\n"
        return attribute_string
