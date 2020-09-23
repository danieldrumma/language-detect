from trainText import trainText
import sys
import math
import json

"""
AI Class 
Author: Daniel Boaitey
Detect if text is English or Dutch with decision tree or adaboost


note that ada hypothesis out will be a json file
train <examples> <hypothesisOut> <learning-type>  
predict <hypothesis> <file>

-------------INSTRUCTIONS FOR EVALUATION DATA--------------------:
enter command line input as 

a - file to train on   ---another_train.dat
b - training file for test file ...this is to compare the test file once it has recieved predictions from the hypothesis
    trained on data from file a   -----train_.dat
c - test file - used to test hypothesis trained from file a   ----test_.dat
d - algorithm to run ---> "dt" or "ada"
a         b          c         d
train.dat train_.dat test_.dat dt
or
train.dat train_.dat test.dat ada

Modify max depth and number of stumps with macros below

"""
# max depth for decision tree
MAX_DEPTH = 5

# number of iterations - ensemble size
ADA_ITERATIONS = 6

# attributes
ATTRIBUTES = ["zijn_presence", "niet_presence", "en_presence",
              "the_presence", "ee_presence", "aa_presence",
              "de_presence", "engl_length", "nl_length", "het_presence", "vowel_comp"]


def parse_training_data(filename):
    """
    Recieve filename and open, create trainText objects for training data
    :param filename:
    :return:
    """
    corpus = []
    with open(filename, 'r', encoding="utf8") as fl:
        lines = len(fl.readlines())
        fl.seek(0)
        wt = 1 / lines
        obj = None
        for line in fl:
            if line.startswith("en"):
                line = line[3:]
                obj = trainText("en", line, wt)
                set_attributes(obj)
            else:
                line = line[3:]
                obj = trainText("nl", line, wt)
                set_attributes(obj)
            corpus.append(obj)
    return corpus


def parse_training_data_pred(filename):
    """
    recieve filename and open, create trainText objects for test data
    :param filename:
    :return:
    """
    corpus = []
    with open(filename, 'r', encoding="utf8") as fl:
        lines = 800
        wt = 1 / lines
        for line in fl:
            obj = trainText("", line, wt)
            set_attributes(obj)
            corpus.append(obj)
    return corpus


def substring_check(substring, text):
    """
    Check if a substring is within a piece of training text
    :param substring: substring to check
    :param text: training test
    :return: boolean - generally True is nl, False is English
    """
    if substring in text.lower():
        return True
    else:
        return False


def ends_check(substring, text):
    """
    Check if any word within training text ends with the given substring
    :param substring: substring to check
    :param text: training text
    :return: boolean - generally True is nl, False is English
    """
    words = text.split(' ')
    to_return = False
    for word in words:
        if word.lower().endswith(substring):
            to_return = True
            break
    return to_return


def numeric_check(threshold, direction, text):
    """
    Check average word length based on threshold
    :param threshold: number to indicate where divide occurs
    :param direction: greater than or less than
    :param text: text for training
    :return:
    """
    words = text.split(' ')
    tot_len = 0
    to_return = False
    for word in words:
        tot_len += len(word)
    avg = tot_len / 15
    if direction == "<":
        if avg < threshold:
            to_return = True
    else:
        if avg > threshold:
            to_return = True
    return to_return


def vowel_check(text):
    """
    Checks vowel composition of text and returns value to indicate if vowel composition is above threshold

    :param text: string to check
    :return: boolean indicating if vowel composition is above threshold
    """
    vcount = 0
    textcnt = 0

    for word in text:
        for vowel in "aeiou":
            count = word.count(vowel)
            vcount += count
        textcnt += len(word)

    avg_vowel_presence = vcount / textcnt
    if avg_vowel_presence > .32:
        return True
    return False


def set_attributes(tText):
    """
    Set attributes for given text
    :param tText: train text object
    :return: none, attribute in tText object changed
    """
    # 15 word string
    text = tText.text

    # presence of zijn yes nl
    zijn_presence = substring_check("zijn", text)
    tText.attributes["zijn_presence"] = zijn_presence

    # presence of niet yes nl ... CHANGE THIS
    niet_presence = substring_check("niet", text)
    # if niet_presence:
    #     niet_presence = False
    # else:
    #     niet_presence = True
    tText.attributes["niet_presence"] = niet_presence

    # words that end with en
    en_presence = ends_check("en", text)
    tText.attributes["en_presence"] = en_presence

    # the within string
    the_presence = substring_check(" the ", text)
    if the_presence:
        the_presence = False
    else:
        the_presence = True
    tText.attributes["the_presence"] = the_presence

    # contains ee
    ee_presence = substring_check("ee", text)
    tText.attributes["ee_presence"] = ee_presence

    # contains aa
    aa_presence = substring_check("aa", text)
    tText.attributes["aa_presence"] = aa_presence

    # contains de
    de_presence = substring_check("de", text)
    tText.attributes["de_presence"] = de_presence

    # avg word leng less 6
    engl_length = numeric_check(6, "<", text)
    if engl_length:
        engl_length = False
    else:
        engl_length = True
    tText.attributes["engl_length"] = engl_length

    # # avg word leng great 6
    nl_length = numeric_check(7.5, ">", text)
    tText.attributes["nl_length"] = nl_length

    # het presence
    het_presence = substring_check("het", text)
    tText.attributes["het_presence"] = het_presence

    # vowel composition
    vowel_comp = vowel_check(text)
    tText.attributes["vowel_comp"] = vowel_comp


def split(current_attribute, corpus):
    """
    Split up corpus based on given attribute
    :param current_attribute: attribute for current run
    :param corpus: trainText list
    :return: two groups, one of all true, one of all false from corpus
    """
    # print("Splitting \n\n")
    group_true = []
    group_false = []
    check = False
    if current_attribute == "":
        check = True
    for tText in corpus:
        if check:
            print(tText)
        if tText.attributes[current_attribute]:
            # print("Ok value is ", tText.attributes[current_attribute])
            group_true.append(tText)
        else:
            # print("Ok value is ", tText.attributes[current_attribute])
            group_false.append(tText)
    # print("Grouping: ", len(group_true), len(group_false))
    return [group_true, group_false]


def find_majority_lang(group):
    """
    Count presence of each language in group, return whichever is majority
    :param group: list of trainText objects
    :return: returns majority language
    """
    en_ct = 0
    nl_ct = 0
    for i in range(0, len(group)):
        if group[i].text == "en":
            en_ct += 1
        else:
            nl_ct += 1
    if en_ct > nl_ct:
        return "en"
    else:
        return "nl"


def missclass_rate(children):
    """
    Used for adaboost
    Calculated weighted missclassification rate
    sums all weights, sums all missclassified weights
    returns sum of all weights/
    :param children:
    :return: sum of misclassified weights / sum of all weights
    """
    wrong_vals = 0
    wt_total = 0
    for i in children:
        for tText in i:
            wt_total += tText.weight
    for i in range(0, 2):
        if i == 0:
            for tText in children[i]:
                if tText.lang != "nl":
                    wrong_vals += tText.weight
        else:
            for tText in children[i]:
                if tText.lang != "en":
                    wrong_vals += tText.weight
    try:
        val = wrong_vals / wt_total
        return val
    except ZeroDivisionError:
        return 0


def entropy(groups):
    """
    Info gain/Entropy calculation for boolean attributes
    entropy that remains after split on current attribute
    
    :param groups: 2 nodes = each contain set of trainText objects
    :return: entropy value for current set of nodes
    """
    true_size = len(groups[0])
    false_size = len(groups[1])
    tot_len = true_size + false_size

    group_true_portion = true_size / tot_len
    group_false_portion = false_size / tot_len

    # avoid division by 0 error
    en_in_true = 1
    nl_in_true = 1
    en_in_false = 1
    nl_in_false = 1

    for tText in groups[0]:
        if tText.lang == "en":
            en_in_true += 1
        else:
            nl_in_true += 1

    for tText in groups[1]:
        if tText.lang == "en":
            en_in_false += 1
        else:
            nl_in_false += 1

    subset_true_entropy = group_true_portion * (
            ((en_in_true / (true_size + 1)) * math.log2(2 / (en_in_true / (true_size + 1)))) +
            ((nl_in_true / (true_size + 1)) * math.log2(2 / (nl_in_true / (true_size + 1)))))

    subset_false_entropy = group_false_portion * (
            ((en_in_false / (false_size + 1)) * math.log2(2 / (en_in_false / (false_size + 1)))) +
            ((nl_in_false / (false_size + 1)) * math.log2(2 / (nl_in_false / (true_size + 1)))))

    return subset_false_entropy + subset_true_entropy


def try_curr_attributes(attributes, corpus, algorithm):
    """
    Find most ideal attribute for creating a new branching in
    decision tree
    :param attributes: list of attributes as stirngs
    :param corpus: list of trainText objects
    :param algorithm: indicate whether adaboost or decision tree
    :return: dictionary with attribute title and access to two groups of children that show split of data 
            based on selected attribute
    """
    best_entropy = sys.maxsize * 2 + 1
    best_groups = []
    ideal_attribute = ""
    for i in range(0, len(attributes)):
        # for i in range(0, 1):
        # print("Checking attribute: ", attributes[i])
        groups = split(attributes[i], corpus)
        if algorithm is "ADA":
            # for adaboost, entropy is not used...
            error = missclass_rate(groups)
            return [error, {"Attribute": attributes[i], "Children": groups}]
        curr_entropy = entropy(groups)
        if curr_entropy < best_entropy:
            ideal_attribute = attributes[i]
            best_entropy = curr_entropy
            best_groups = groups
    return {"Attribute": ideal_attribute, "Children": best_groups}


def check_lang_same(dt):
    """
    check if all values in the data have the same language
    :param dt: list of trainText objects
    :return: boolean value
    """
    allvals = len(dt)
    lang = ""
    en_ct = 0
    nl_ct = 0
    for i in dt:
        if i.lang == "en":
            en_ct += 1
        if i.lang == "nl":
            nl_ct += 1
    return nl_ct == allvals or en_ct == allvals


def decision_tree(init, attributes, corpus, root, depth, parent, algorithm):
    """
    Builds a decision tree, checks for stopping conditions
    recurses on left/right nodes

    :param init: if True, this indicates the first run for building a decision tree 
                        OR
                indicates adaboost run
    :param attributes: list of attributes as strings
    :param corpus: list of trainText objects
    :param root: initially None, is a dictionary object in every other instance
    :param depth: current depth of tree
    :param parent: parent of current root
    :param algorithm: DT means normal decision tree, whereas ADA is adaboost
    :return:
    """
    if algorithm is "DT":
        if check_lang_same(corpus):
            del (root['Children'])
            root["Class"] = corpus[0].lang
            return root
        elif len(attributes) == 0:
            del (root['Children'])
            # print("Ended at none features, printing parent")
            return parent
        elif len(attributes) == 1:
            del (root['Children'])
            # print("Ended at none features, printing parent b")
            return parent

    if init and algorithm is "DT":
        root = try_curr_attributes(attributes, corpus, algorithm)
    elif init and algorithm is "ADA":
        # single stump built for adaboost
        result = try_curr_attributes(attributes, corpus, algorithm)
        return result

    attributes.remove(root["Attribute"])

    left_true_child = root["Children"][0]
    right_false_child = root["Children"][1]
    maj_lang = find_majority_lang(root["Children"][0])
    del (root['Children'])  # values of children are not needed beyond this point in the root structure
    # print("Depth is ", depth)

    if len(right_false_child) == 0 or len(left_true_child) == 0:
        root['Class'] = maj_lang
        # print("SHOULD BE DONE b")

    if depth >= MAX_DEPTH:
        # print("Ended in here")
        # root['Left'] = maj_lang
        # root['Right'] = maj_lang
        root["Class"] = maj_lang
        # pprint.pprint(root)
        return root
    else:
        if len(left_true_child) > 6:
            root['Left'] = try_curr_attributes(attributes, left_true_child, algorithm)
            decision_tree(False, attributes, left_true_child, root['Left'], depth + 1, root, algorithm)
        if len(right_false_child) > 6:
            root['Right'] = try_curr_attributes(attributes, right_false_child, algorithm)
            decision_tree(False, attributes, right_false_child, root['Right'], depth + 1, root, algorithm)
    return root


def to_predict(tree, data):
    """
    Recursive function to use tree to predict language of given text
    :param tree: decision tree
    :param data: trainText object
    :return:
    """
    val = tree["Attribute"]
    curr_val = data.attributes[val]
    to_return = None
    if curr_val:
        if "Left" in tree:
            to_return = to_predict(tree["Left"], data)
        elif "Class" in tree:
            to_return = tree["Class"]
        else:
            to_return = "nl"
    else:
        if "Right" in tree:
            to_return = to_predict(tree["Right"], data)
        elif "Class" in tree:
            to_return = tree["Class"]
        else:
            to_return = "en"
    return to_return


def dt_predict(root, test):
    """
    coordinates prediction for all values in test corpus
    applies tree to each value and records data appropriately
    :param root: decision tree to detect language
    :param test: set of test data - list of trainText objects that have no lang value
    """
    for i in test:
        predict = to_predict(root, i)
        print(predict)


def dt_predict_eval(root, test, train):
    """
    coordinates prediction for all values in test corpus
    applies tree to each value and records data appropriately
    :param root: decision tree to detect language
    :param test: set of test data - list of trainText objects that have no lang value
    """
    lstp = []
    for i in test:
        predict = to_predict(root, i)
        i.lang = predict
        lstp.append(predict)
    nlc = 0
    enc = 0
    for i in lstp:
        if i == "en":
            enc += 1
        else:
            nlc += 1

    total_right = 0
    total = len(test)
    total_wrong = 0
    en_wrong = 0
    nl_wrong = 0
    all_nl = 0
    all_en = 0
    test_en = 0
    test_nl = 0
    for a, b in zip(test, train):
        if b.lang == "en":
            all_en += 1
        else:
            all_nl += 1

        if a.lang == "en":
            test_en += 1
        else:
            test_nl += 1
        if a.lang == b.lang:
            total_right += 1
        else:
            if b.lang == "en":
                nl_wrong += 1
            else:
                en_wrong += 1
            total_wrong += 1
    print("The accuracy of this run is " + str((total_right / total) * 100) + " %. " + str(
        total_right) + " were correct of "
                       "the " +
          str(total))
    print("The error rate is " + str((total_wrong / total) * 100) + " %. " + str(total_wrong)
          + " were incorrect of the " + str(total))
    print("\n//////////////////////////////////////////////")
    print("Total English in training file is ", all_en)
    print("Total Dutch in training file is ", all_nl)
    print("Total English predicted in test file is ", test_en)
    print("Total Dutch predicted in test file is ", test_nl)
    print("\n//////////////////////////////////////////////")
    print(str(nl_wrong) + " were incorrectly classified as English.")
    print(str(en_wrong) + " were incorrectly classified as Dutch.")


def stump_selector(stumps):
    """
    Pick stump with lowest error

    :return: best stump
    """
    min_error = sys.maxsize * 2 + 1
    best_stump = None
    for stump in stumps:
        if stump[0] < min_error:
            min_error = stump[0]
            best_stump = stump
    # print("Stump selected has attribute ", best_stump[1]["Attribute"], " error is", min_error)
    return best_stump


def get_alpha(stump):
    """
    calculate alpha for current stump
    output weight for this classifier essentially
    :param stump: stump
    :return: alpha value
    """
    error = stump[0]
    error = error

    try:
        return .5 * (math.log((1 - error) / error))
    except ZeroDivisionError:
        print("Zero division occurance")
        return 2.5
    except ValueError as e:
        print(1 - error, error, e)
        return 2.5


def find_misclassified(children):
    """
    isolate which data items have been misclassified
    :param children: left and right node data of best stump
    :return: list misclassified IDs
    """
    missclass_ids = []
    for i in range(1):
        if i == 0:
            for tText in children[i]:
                if tText.lang != "nl":
                    missclass_ids.append(tText.id)
        else:
            for tText in children[i]:
                if tText.lang != "en":
                    missclass_ids.append(tText.id)
    return missclass_ids


def set_new_weights(corpus, alpha, wt_diff_lst):
    """
    create new sample weights and normalize
    :param corpus: list of trainText objects
    :param alpha: alpha, score of current ideal stump
    :return:
    """
    total = 0
    for tText in corpus:
        if tText.id in wt_diff_lst:
            tru_alpha = alpha
        else:
            tru_alpha = -1 * alpha
        # print("wt before ", tText.weight)
        tText.weight = tText.weight * math.exp(tru_alpha)
        total += tText.weight
    #  print("wt after ", tText.weight)

    # print("Un-normalized weight", total)
    new_tot = 0
    for tText in corpus:
        tText.weight = tText.weight / total
        new_tot += tText.weight
    # print("New total is ", new_tot)


# def new_data(corpus):
#     """
#      not in use
#      intended for rebuilding new data for stump selection
#     :param corpus:
#     :return:
#     """
#     fresh_corpus = []
#     bottom_lim = 0
#    # print(len(corpus))
#     upper_lim = corpus[0].weight
#     lims = []
#     corpus_size = len(corpus)
#     if corpus_size < 100:
#         print("Short one", upper_lim)
#     else:
#         print("NOt short", upper_lim)
#         #print(corpus[0])
#     for i in range(0, corpus_size):
#         lims.append([bottom_lim, upper_lim])
#         # lims.append([bottom_lim, upper_lim])
#         if corpus_size < 100:
#             print(lims[i])
#         bottom_lim = upper_lim
#         upper_lim += corpus[i].weight
#
#     # resample data
#     print("NEW GAME ", lims)
#     for i in range(0, corpus_size):
#         val = random.uniform(0, 1)
#         for j in range(0, len(lims)):
#             #            if val in drange(Decimal(lims[j][0]), Decimal(lims[j][1]), '.000000001'):
#             if lims[j][0] <= val < lims[j][1]:
#                 # corpus[j].tText = 1 / (corpus_size)
#                 fresh_corpus.append(corpus[j])
#                 break
#     print("Fresh", len(fresh_corpus))
#     print("post ", fresh_corpus[0].weight)
#     return fresh_corpus


def adaboost_runner(attributes, corpus, iterations):
    """
    Main control function for adaboost.
    Sets up loop for given amount of iterations to create ensemble
    Output of each stump with alpha value as dict in list
    :param attributes: list of attributes as strings
    :param corpus: list of trainText objects
    :param iterations: number of ensemble items to yield
    :return:
    """
    ensemble = []
    for run in range(0, iterations):
        curr_stumps = []
        for attr in attributes:
            attribute_as_lst = [attr]
            tree = None
            stump_info = decision_tree(True, attribute_as_lst, corpus, tree, 1, None, "ADA")
            curr_stumps.append(stump_info)

        # select stump with smallest error
        best_stump = stump_selector(curr_stumps)

        # get alpha for current stump
        alpha = get_alpha(best_stump)

        # add stump to ensemble
        ensemble.append([best_stump, alpha])

        # get all missclassified nodes
        wt_diff_lst = find_misclassified(best_stump[1]["Children"])
        # print(wt_diff_lst)

        # recalculate weights of corpus
        set_new_weights(corpus, alpha, wt_diff_lst)

    out_struct = []
    for i in ensemble:
        tree = i[0][1]
        tree["Alpha"] = i[1]
        del tree["Children"]
        out_struct.append(tree)

    return out_struct


def adaboost_predictor(corpus, ensemble):
    """
    Run prediction for items with adaboost ensemble.
    :param corpus: list of trainText objects
    :param ensemble: used to form prediction, result of adaboost_runner
    :return:
    """
    num_nl = 0
    num_en = 0
    for val in corpus:
        for tree in ensemble:
            score = tree["Alpha"]
            attr = tree["Attribute"]
            value = val.attributes[attr]
            if value is True:
                num_nl += score
            else:
                num_en += score
        if num_nl > num_en:
            print("nl")
        else:
            print("en")
        num_nl = 0
        num_en = 0


def adaboost_predictor_eval(test, train, ensemble):
    """
    Run prediction for items with adaboost ensemble.
    :param corpus: list of trainText objects
    :param ensemble: used to form prediction, result of adaboost_runner
    :return:
    """
    num_nl = 0
    num_en = 0
    total_right = 0
    total = len(test)
    total_wrong = 0
    en_wrong = 0
    nl_wrong = 0
    all_nl = 0
    all_en = 0
    test_en = 0
    test_nl = 0
    for texta, textb in zip(test, train):
        for tree in ensemble:
            score = tree["Alpha"]
            attr = tree["Attribute"]
            value = texta.attributes[attr]
            if value is True:
                num_nl += score
            else:
                num_en += score
        if num_nl > num_en:
            test_nl += 1
            texta.lang = "nl"
        else:
            test_en += 1
            texta.lang = "en"
        if textb.lang == "en":
            all_en += 1
        else:
            all_nl += 1
        if texta.lang == textb.lang:
            total_right += 1
        else:
            if textb.lang == "en":
                nl_wrong += 1
            else:
                en_wrong += 1
            total_wrong += 1

        num_nl = 0
        num_en = 0

    print("The accuracy of this run is " + str((total_right / total) * 100) + " %. " + str(
        total_right) + " were correct of "
                       "the " +
          str(total))
    print("The error rate is " + str((total_wrong / total) * 100) + " %. " + str(total_wrong)
          + " were incorrect of the " + str(total))
    print("\n//////////////////////////////////////////////")
    print("Total English in training file is ", all_en)
    print("Total Dutch in training file is ", all_nl)

    print("Total English predicted in test file is ", test_en)
    print("Total Dutch predicted in test file is ", test_nl)
    print("\n//////////////////////////////////////////////")
    print(str(nl_wrong) + " were incorrectly classified as English.")
    print(str(en_wrong) + " were incorrectly classified as Dutch.")


def main_train(examples, hypothesis_name, alg):
    """
    main function for training - sends data to decision tree or adaboost depending on alg value
    :return:
    """
    corpus = parse_training_data(examples)
    tree = None

    if alg == 'dt':
        a_tree = decision_tree(True, ATTRIBUTES, corpus, tree, 1, None, "DT")
        marker = {"dt": a_tree}
        with open(hypothesis_name + '.json', 'w') as f:
            json.dump(marker, f)

    else:
        ensemble = adaboost_runner(ATTRIBUTES, corpus, ADA_ITERATIONS)
        with open(hypothesis_name + '.json', 'w') as f:
            json.dump(ensemble, f)


def main_predict(hypothesis, file):
    """
    main funciton for prediction, sends test data to decision tree predictor or
    adaboost predictor depending on type of file
    :return:
    """
    test = parse_training_data_pred(file)

    with open(hypothesis + '.json') as f:
        hypothesis = json.load(f)

    if type(hypothesis) is dict:
        dt_predict(hypothesis["dt"], test)
    else:
        adaboost_predictor(test, hypothesis)


def main_evaluate(train, test_label, test_pure, algorithm):
    """
    Evaluate accuracy.
    :param train: file for training
    :param test_label: file to check values after prediction
    :param test_pure: file to test hypothesis on
    :param algorithm: indicate adaboost or decision tree
    """
    train_corpus = parse_training_data(train)
    test_labeled = parse_training_data(test_label)
    test_corpus = parse_training_data_pred(test_pure)

    for tText, tTextb in zip(train_corpus, test_corpus):
        tTextb.id = tText.id

    if algorithm == 'dt':
        print("Results processing for decision tree...")
        # train
        a_tree = decision_tree(True, ATTRIBUTES, train_corpus, None, 1, None, "DT")
        marker = {"dt": a_tree}
        with open('hypothesis.json', 'w') as f:
            json.dump(marker, f)
        f.close()
        # predict
        with open('hypothesis.json') as f:
            hypothesis = json.load(f)
        f.close()
        if type(hypothesis) is dict:
            dt_predict_eval(hypothesis["dt"], test_corpus, test_labeled)
    else:
        print("Results processing for ada boosting...")

        # train
        ensemble = adaboost_runner(ATTRIBUTES, train_corpus, ADA_ITERATIONS)
        with open('hypothesis.json', 'w') as f:
            json.dump(ensemble, f)
        f.close()

        # predict
        with open('hypothesis.json') as f:
            hypothesis = json.load(f)
        f.close()

        adaboost_predictor_eval(test_corpus, test_labeled, hypothesis)


if __name__ == "__main__":
    task = sys.argv[1]
    if len(sys.argv) == 5 and task == "train":
        examples = sys.argv[2]
        hypothesis_name = sys.argv[3]
        algorithm = sys.argv[4]
        main_train(examples, hypothesis_name, algorithm)
    elif len(sys.argv) == 4 and task == "predict":
        hypothesis = sys.argv[2]
        file = sys.argv[3]
        main_predict(hypothesis, file)
    else:
        main_evaluate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
