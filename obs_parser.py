# frequency = 20 Hz
# time: milliseconds

input_filename = 'obs_log06.txt'
output_filename = 'obs06.txt'

f = open(input_filename, 'r')
f_out = open(output_filename, 'w')

prev_left = 0
prev_right = 0

while True:
    if not f.readline().split():
        break
    _ = f.readline()
    sec = f.readline().split()[-1]
    nanoseq = f.readline().split()[-1]
    _ = f.readline()
    left = int(f.readline().split()[-1])
    right = int(f.readline().split()[-1])
    _ = f.readline()

    sec = sec + nanoseq[:-3].zfill(3)
    if left - prev_left > 4e9:
        left -= pow(2, 32)
    if right - prev_right > 4e9:
        right -= pow(2, 32)
    
    prev_left = left
    prev_right = right
    
    f_out.write(' '.join([sec, str(left), str(right)]) + '\n')

f.close()
f_out.close()
