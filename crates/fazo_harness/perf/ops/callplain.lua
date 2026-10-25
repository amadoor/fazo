local x; local s = string.byte; local str = 'abc'
for i = 1, 200000 do
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
    x = s(str)
end
