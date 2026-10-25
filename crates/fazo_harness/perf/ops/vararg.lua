local x; local function f(...) return ... end
for i = 1, 200000 do
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
    x = f(i)
end
