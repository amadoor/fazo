local t = {1, 2, 3}; local k = 2; t[k] = t; local x = t
for i = 1, 200000 do
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
    x = x[k]
end
print(x == t)
