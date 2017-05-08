import pandas
from cStringIO import StringIO
import csv

def read_until(f, line_test, allow_eof = False):
    done = False
    num_lines = 0
    while not done:
        num_lines += 1
        inp = f.readline()
        if inp == '':  # end of file
            if allow_eof:
                done = True
            else:
                return None, inp
        elif line_test(inp): # found the test condition
            done = True
    return num_lines, inp

def read_star_file(fname):
    data_blocks = []
    with open(fname, 'r') as f:
        current_num = 0
        done_reading_data_blocks = False
        while not done_reading_data_blocks:
            num, val = read_until(f, lambda x : x.strip().startswith('data_'))
            if num is None:
                if len(data_blocks) == 0:
                    assert False, "Cannot find any 'data_' blocks in the STAR file %s." % fname
                else:
                    done_reading_data_blocks = True
            else:
                current_num += num
                data_block_start = current_num - 1 # which line is 'data_'

                num, val = read_until(f, lambda x : x.strip().startswith('loop_'))
                assert num is not None, "Cannot find any 'loop_' in the data block starting at line %d in the STAR file." % data_block_start
                current_num += num
                loop_start = current_num - 1

                num, val = read_until(f, lambda x : x.strip().startswith('_'))
                assert num is not None, "Cannot find start of label names in the data block starting at line %d in the STAR file." % data_block_start
                current_num += num
                labels_start = current_num - 1
                labels = []
                while True:
                    val = val.strip()
                    if val.startswith('_'):
                        labels.append(val.lstrip('_').split(' ')[0])
                    elif val.startswith('#') or val.startswith(';'):
                        pass
                    else:
                        break
                    current_num += 1
                    val = f.readline()
                    if val == '':
                        break

                table_start = current_num - 1
                num, val = read_until(f, lambda x : x.strip() == '' , allow_eof = True) # look for empty line or EOF
                current_num += num
                table_end = current_num - 1

                data_blocks.append( (data_block_start, loop_start, labels_start, labels, table_start, table_end)  )

    dataframes = []
    for data_block_start, loop_start, labels_start, labels, table_start, table_end in data_blocks:
        dataframe = pandas.read_csv(fname, names = labels, delim_whitespace = True, 
                                           skiprows = table_start, skip_blank_lines = False,
                                           comment = '#', 
                                           error_bad_lines = True,
                                           nrows = table_end - table_start )
        dataframes.append(dataframe)

    # concat the data blocks only keeping common columns
    dataframe = pandas.concat(dataframes, join='inner', ignore_index=True) 

    return dataframe

def write_star_file(dataframe, fname):
    with open(fname, 'w') as f:
        f.write('\ndata_\n\nloop_\n')
        for cnum, cname in enumerate(dataframe.columns.tolist()):
            f.write('_' + cname + ' #%d\n' % (cnum+1) )
        
        for startidx in range(0, len(dataframe), 1000):
            s = StringIO()
            dataframe.iloc[startidx:startidx+1000].to_csv(s, sep = ';', quoting=csv.QUOTE_NONE, float_format="%20.6f", header=False, index=False)
            f.write(s.getvalue().replace(';', ' '))
