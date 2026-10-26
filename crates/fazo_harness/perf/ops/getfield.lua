local t = {a = 1, b = 2}; t.a = t; local x = t
for i = 1, 200000 do
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
    x = x.a
end
print(x == t)
