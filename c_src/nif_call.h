#ifndef NIF_CALL_H
#define NIF_CALL_H

#pragma once

#include <erl_nif.h>

#ifdef NIF_CALL_NAMESPACE
#define NIF_CALL_CAT(A, B) A##B
#define NIF_CALL_SYMBOL(A, B) NIF_CALL_CAT(A, B)

#define CallbackNifRes NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, CallbackNifRes)
#define nif_call_onload NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, nif_call_onload)
#define prepare_nif_call NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, prepare_nif_call)
#define make_nif_call NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, make_nif_call)
#define nif_call_evaluated NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, nif_call_evaluated)
#define destruct_nif_call_res NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, destruct_nif_call_res)
#endif

#define NIF_CALL_NIF_FUNC(name) \
  {#name, 2, nif_call_evaluated, 0}

#ifndef NIF_CALL_IMPLEMENTATION

struct CallbackNifRes;
static int nif_call_onload(ErlNifEnv *env);
static CallbackNifRes * prepare_nif_call(ErlNifEnv* env);
static ERL_NIF_TERM make_nif_call(ErlNifEnv* caller_env, ErlNifPid evaluator, ERL_NIF_TERM fun, ERL_NIF_TERM args);
static ERL_NIF_TERM nif_call_evaluated(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static void destruct_nif_call_res(ErlNifEnv *, void *obj);

#else

struct CallbackNifRes {
  static ErlNifResourceType *type;
  static ERL_NIF_TERM kAtomNil;
  static ERL_NIF_TERM kAtomENOMEM;

  ErlNifEnv * msg_env;
  ErlNifMutex *mtx = NULL;
  ErlNifCond *cond = NULL;
  
  ERL_NIF_TERM return_value;
  bool return_value_set;
};

ErlNifResourceType * CallbackNifRes::type = NULL;
ERL_NIF_TERM CallbackNifRes::kAtomNil;
ERL_NIF_TERM CallbackNifRes::kAtomENOMEM;

CallbackNifRes * prepare_nif_call(ErlNifEnv* env) {
  CallbackNifRes *res = (CallbackNifRes *)enif_alloc_resource(CallbackNifRes::type, sizeof(CallbackNifRes));
  if (!res) return NULL;
  memset(res, 0, sizeof(CallbackNifRes));

  res->msg_env = enif_alloc_env();
  if (!res->msg_env) {
    enif_release_resource(res);
    return NULL;
  }

  res->mtx = enif_mutex_create((char *)"nif_call_mutex");
  if (!res->mtx) {
    enif_free_env(res->msg_env);
    enif_release_resource(res);
    return NULL;
  }

  res->cond = enif_cond_create((char *)"nif_call_cond");
  if (!res->cond) {
    enif_free_env(res->msg_env);
    enif_mutex_destroy(res->mtx);
    enif_release_resource(res);
    return NULL;
  }

  res->return_value_set = false;
  res->return_value = CallbackNifRes::kAtomNil;

  return res;
}

static ERL_NIF_TERM make_nif_call(ErlNifEnv* caller_env, ErlNifPid evaluator, ERL_NIF_TERM fun, ERL_NIF_TERM args) {
  CallbackNifRes *callback_res = prepare_nif_call(caller_env);
  if (!callback_res) return CallbackNifRes::kAtomENOMEM;

  ERL_NIF_TERM callback_term = enif_make_resource(caller_env, (void *)callback_res);
  enif_send(caller_env, &evaluator, callback_res->msg_env, enif_make_copy(callback_res->msg_env, enif_make_tuple3(caller_env,
    fun,
    args,
    callback_term
  )));

  enif_mutex_lock(callback_res->mtx);
  while (!callback_res->return_value_set) {
    enif_cond_wait(callback_res->cond, callback_res->mtx);
  }
  enif_mutex_unlock(callback_res->mtx);

  ERL_NIF_TERM return_value = enif_make_copy(caller_env, callback_res->return_value);
  enif_release_resource(callback_res);
  
  return return_value;
}

static ERL_NIF_TERM nif_call_evaluated(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  CallbackNifRes *res = NULL;
  if (!enif_get_resource(env, argv[0], CallbackNifRes::type, (void **)&res)) return enif_make_badarg(env);

  res->return_value = enif_make_copy(res->msg_env, argv[1]);
  res->return_value_set = true;
  enif_cond_signal(res->cond);

  return enif_make_atom(env, "ok");
}

static void destruct_nif_call_res(ErlNifEnv *, void *obj) {
  CallbackNifRes *res = (CallbackNifRes *)obj;
  if (res->cond) {
    enif_cond_destroy(res->cond);
    res->cond = NULL;
  }
  if (res->mtx) {
    enif_mutex_destroy(res->mtx);
    res->mtx = NULL;
  }
  if (res->msg_env) {
    enif_free_env(res->msg_env);
    res->msg_env = NULL;
  }
}

static int nif_call_onload(ErlNifEnv *env) {
  static int loaded = 0;
  if (loaded) return 0;

  ErlNifResourceType *rt;
  rt = enif_open_resource_type(env, "Elixir.NifCall.NIF", "CallbackNifRes", destruct_nif_call_res, ERL_NIF_RT_CREATE, NULL);
  if (!rt) return -1;
  CallbackNifRes::type = rt;

  CallbackNifRes::kAtomNil = enif_make_atom(env, "nil");
  CallbackNifRes::kAtomENOMEM = enif_make_atom(env, "enomem");
  loaded = 1;
  return 0;
}

#endif

#endif  // NIF_CALL_H
