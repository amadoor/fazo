local x = 1; local o = {}; function o:m(a) return a end
for i = 1, 200000 do
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
    x = o:m(x)
end
print(x)
