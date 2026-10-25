local x; local function g(a) return a end; local function f(a) return g(a) end
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
