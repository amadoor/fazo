local t = {1, 2, 3}; t[2] = t; local x = t
for i = 1, 200000 do
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
    x = x[2]
end
print(x == t)
