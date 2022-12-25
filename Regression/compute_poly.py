lst = []

def CompPoly(a, n):
    if(n > 1):
        total = 0
        if(len(a) > 1):
            for i in range(0, n + 1):
                cur_val = pow(a[0], n - i) * CompPoly(a[1:], i)
                total += cur_val
                lst.append(cur_val)
        else:
            return pow(a[0], n)
        return total
    if(n==1):
        # lst.extend(a)
        return sum(a)
    return 1

if __name__ == "__main__":
    CompPoly([3, 4, 5, 6] , 2)
    print(lst)