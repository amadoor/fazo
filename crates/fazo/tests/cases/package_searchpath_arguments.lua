print(pcall(package.searchpath))
print(pcall(package.searchpath, "name"))
print(package.searchpath("no_such_module_q", "./?.lua;./?/init.lua") == nil)
