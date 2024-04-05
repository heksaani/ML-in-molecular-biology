def convert_div(s):
    div_open = False
    par_o = 0
    par_c = 0
    i = 0
    ft = ""
    st = ""
    mult = ""
    ft_start = 0
    st_start = 0

    while True:
        if i == len(s):
            break

        if s[i:i+3] == "div" and not div_open:
            div_open = True
            i += 4
            ft_start = i
            start = i

        if div_open:
            if s[i] == "(":
                par_o += 1

            if s[i] == ")":
                par_c += 1

            if s[i] == "," and par_o == par_c:
                mult = "Mul(" + s[ft_start:i] + ", Pow(" + s[i+2:len(s)-1] + ", -1))"
                break

        i += 1
    return mult

def find_div(s):
    start = s.find("div")
    i = start
    end = 0
    par_o = 0
    par_c = 0
    while True:
        if s[i] == "(":
            par_o += 1
        elif s[i] == ")":
            par_c += 1

        if par_o == par_c and par_o != 0:
            end = i
            break
        i += 1
    return start, end

def convert_sub(s):
    sub_open = False
    par_o = 0
    par_c = 0
    i = 0
    ft = ""
    st = ""
    add = ""
    ft_start = 0
    st_start = 0

    while True:
        if i == len(s):
            break

        if s[i:i+3] == "sub" and not sub_open:
            # print("Open")
            sub_open = True
            i += 4
            ft_start = i
            start = i

        if sub_open:
            if s[i] == "(":
                par_o += 1

            if s[i] == ")":
                par_c += 1

            if s[i] == "," and par_o == par_c:
                add = "Add(" + s[ft_start:i] + ", -" + s[i+2:len(s)-1] + ")"
                break

        i += 1
    return add

def find_sub(s):
    start = s.find("sub")
    i = start
    end = 0
    par_o = 0
    par_c = 0
    while True:
        if s[i] == "(":
            par_o += 1
        elif s[i] == ")":
            par_c += 1

        if par_o == par_c and par_o != 0:
            end = i
            break
        i += 1
    return start, end

def convert_s(s):
    while "div" in s:
        start, end = find_div(s)
        s = s[:start] + convert_div(s[start:end+1]) + s[end+1:]

    while "sub" in s:
        start, end = find_sub(s)
        s = s[:start] + convert_sub(s[start:end+1]) + s[end+1:]

    s = s.replace("mul", "Mul")
    s = s.replace("add", "Add")
    return s