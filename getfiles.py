import glob
import paths
import config


def getfiles():
    train_dev_boundary = -1
    files = glob.glob(paths.path2KTC + 'syn/*.*')

    if config.CABOCHA_SPLIT:
        files = glob.glob(paths.path2KTC + 'syn/95010[1-9].*')
        train_dev_boundary = -1
    best_acc = 0.0
    least_loss = 1000.0
    update = False
    global early_stop_count
    early_stop_count = 0

    # files = [paths.path2KTC + 'syn/9501ED.KNP', paths.path2KTC + 'syn/9501ED.KNP']

    if config.STANDARD_SPLIT:
        files = glob.glob(paths.path2KTC + 'syn/95010[1-9].*')
        files.extend(glob.glob(paths.path2KTC + 'syn/95011[0-1].*'))
        files.extend(glob.glob(paths.path2KTC + 'syn/950[1-8]ED.*'))
        if config.TEST:
            files.extend(glob.glob(paths.path2KTC + 'syn/95011[4-7].*'))
            files.extend(glob.glob(paths.path2KTC + 'syn/951[0-2]ED.*'))
            train_dev_boundary = -7
        else:
            files.extend(glob.glob(paths.path2KTC + 'syn/95011[2-3].*'))
            files.extend(glob.glob(paths.path2KTC + 'syn/9509ED.*'))
            train_dev_boundary = -3

    if config.JOS:
        files = glob.glob(paths.path2KTC + 'just-one-sentence.txt')
        files = [paths.path2KTC + 'just-one-sentence.txt', paths.path2KTC + 'just-one-sentence.txt']

    if config.MINI_SET:
        files = [paths.path2KTC + 'miniKTC_train.txt', paths.path2KTC + 'miniKTC_dev.txt']

    print(files)

    return files, train_dev_boundary
