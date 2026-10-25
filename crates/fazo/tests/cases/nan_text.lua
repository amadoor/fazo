print(0/0, -(0/0), tostring(0/0), tostring(-(0/0)))
print(string.format("%.17g|%5.1f|%G|%E|%+f", 0/0, -(0/0), -(0/0), 0/0, 0/0))
print(string.format("%a|%A|%a", 5e-324, 2.5e-310, 2.5e-310))
print(string.format("%g|%f", 1/0, -1/0), tostring(1/0), tostring(-1/0))
