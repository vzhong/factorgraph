package = "factorgraph"
version = "scm-1"

source = {
   url = "git://github.com/vzhong/factorgraph",
   tag = "master"
}

description = {
  summary = "Factor graph and belief propagation in Lua",
  detailed = [[
    Factor graph and (loopy) belief probagation in Lua.
  ]],
  homepage = "https://github.com/vzhong/factorgraph"
}

dependencies = {
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
