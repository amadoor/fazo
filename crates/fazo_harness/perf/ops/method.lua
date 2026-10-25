local x; local o = {}; function o:m(a) return a end
for i = 1, 200000 do
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
    x = o:m(i)
end
