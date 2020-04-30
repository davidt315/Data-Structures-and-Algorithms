
def print_table(table):
    for i in range(1, len(table)):
        s = ""
        for j in range(1, len(table[0])):
            s = s + str(table[i][j]) + "    "
        print(s)



def wildcard(s1, s2):
    """
    Wildcard takes in two strings where:
    s1 = pattern string
    s2 = wildcard string

    and returns whether or not there is a solution to the wildcard problem
    """
    # initialization or truth matrix
    table = [[False for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    table[0][0] = True

    # error with the matrix not being able to check if first instance is "*"
    if(s2[0] == "*"):
        table[1][0] = True


    for n in range(1,len(s1)+1):
        for m in range(1,len(s2)+1):
            # condition where the two characters are the same
            if(s1[n-1] == s2[m-1]):
                table[n][m] = table[n-1][m-1]
            # condition where the wildcard string's current value is "*"
            elif(s2[m-1] == "*"):
                table[n][m] = table[n][m-1] or table[n-1][m]
    print()
    print_table(table)
    return table[-1][-1]

s1 = "abcdefg"
s2 = "ab*defg"
print(wildcard(s1,s2))

s1 = "abcdefg"
s2 = "ab*cdefg"
print(wildcard(s1,s2))

s1 = "abcdefg"
s2 = "ab*fg"
print(wildcard(s1,s2))

s1 = "abcdefg"
s2 = "ab*de*g"
print(wildcard(s1,s2))

s1 = "abcdefg"
s2 = "abcdef*"
print(wildcard(s1,s2))

s1 = "abcdefg"
s2 = "*bcdefg"
print(wildcard(s1,s2))