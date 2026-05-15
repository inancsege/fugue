package fugue

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// mustReflect runs reflectInputSchema on sample's type and returns the
// resulting JSON Schema RawMessage. Fails the test on error.
func mustReflect(t *testing.T, sample any) json.RawMessage {
	t.Helper()
	got, err := reflectInputSchema(reflect.TypeOf(sample))
	if err != nil {
		t.Fatalf("reflectInputSchema: %v", err)
	}
	return got
}

func TestReflect_StringField(t *testing.T) {
	type In struct {
		Q string `json:"q" fugue:"the search query"`
	}
	got := mustReflect(t, In{})
	want := `{"type":"object","properties":{"q":{"description":"the search query","type":"string"}},"required":["q"]}`
	if string(got) != want {
		t.Errorf("schema =\n%s\nwant\n%s", got, want)
	}
}

func TestReflect_BoolField(t *testing.T) {
	type In struct {
		Active bool `json:"active"`
	}
	got := mustReflect(t, In{})
	want := `{"type":"object","properties":{"active":{"type":"boolean"}},"required":["active"]}`
	if string(got) != want {
		t.Errorf("schema =\n%s\nwant\n%s", got, want)
	}
}

func TestReflect_IntegerFields(t *testing.T) {
	type In struct {
		A int   `json:"a"`
		B int8  `json:"b"`
		C int64 `json:"c"`
		D uint8 `json:"d"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"a":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"b":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"c":{"type":"integer"}`) ||
		!strings.Contains(string(got), `"d":{"type":"integer"}`) {
		t.Errorf("expected all integer types, got: %s", got)
	}
}

func TestReflect_FloatFields(t *testing.T) {
	type In struct {
		X float32 `json:"x"`
		Y float64 `json:"y"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"x":{"type":"number"}`) ||
		!strings.Contains(string(got), `"y":{"type":"number"}`) {
		t.Errorf("expected float types as number, got: %s", got)
	}
}

func TestReflect_BareUintField(t *testing.T) {
	type In struct {
		Count uint `json:"count"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"count":{"type":"integer"}`) {
		t.Errorf("expected count:{type:integer}, got: %s", got)
	}
}

func TestReflect_NonStructTopLevelRejected(t *testing.T) {
	_, err := reflectInputSchema(reflect.TypeOf("hello"))
	if err == nil {
		t.Fatal("expected error for non-struct top-level type")
	}
	if !strings.Contains(err.Error(), "struct") {
		t.Errorf("error should mention struct, got: %v", err)
	}
}

func TestReflect_SliceOfString(t *testing.T) {
	type In struct {
		Tags []string `json:"tags"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"tags":{"type":"array","items":{"type":"string"}}`) {
		t.Errorf("schema = %s", got)
	}
}

func TestReflect_MapStringToInt(t *testing.T) {
	type In struct {
		Counts map[string]int `json:"counts"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"counts":{"type":"object","additionalProperties":{"type":"integer"}}`) {
		t.Errorf("schema = %s", got)
	}
}

func TestReflect_PointerFieldNotRequired(t *testing.T) {
	type In struct {
		Optional *string `json:"opt"`
		Required string  `json:"req"`
	}
	got := mustReflect(t, In{})
	// "opt" should NOT appear in required; "req" should.
	if strings.Contains(string(got), `"required":["opt"`) || strings.Contains(string(got), `"opt","req"`) {
		t.Errorf("pointer field opt should not be required: %s", got)
	}
	if !strings.Contains(string(got), `"required":["req"]`) {
		t.Errorf("required should be [req], got: %s", got)
	}
	// "opt" should still be in properties with the underlying type schema.
	if !strings.Contains(string(got), `"opt":{"type":"string"}`) {
		t.Errorf("opt property missing or wrong: %s", got)
	}
}

func TestReflect_OmitemptyDropsRequired(t *testing.T) {
	type In struct {
		Limit int `json:"limit,omitempty"`
	}
	got := mustReflect(t, In{})
	if strings.Contains(string(got), `"required"`) {
		t.Errorf("omitempty field should not appear in required: %s", got)
	}
}

func TestReflect_JSONRawMessageIsAny(t *testing.T) {
	type In struct {
		Blob json.RawMessage `json:"blob"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"blob":{}`) {
		t.Errorf("expected blob:{}, got: %s", got)
	}
}

func TestReflect_NestedStruct(t *testing.T) {
	type Inner struct {
		X int `json:"x"`
	}
	type In struct {
		I Inner `json:"i"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"i":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}`) {
		t.Errorf("schema = %s", got)
	}
}

func TestReflect_FugueEnum(t *testing.T) {
	type In struct {
		Mode string `json:"mode" fugueEnum:"semantic,exact,fuzzy"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"mode":{"enum":["semantic","exact","fuzzy"],"type":"string"}`) {
		t.Errorf("schema = %s", got)
	}
}

func TestReflect_FugueEnumWithDescription(t *testing.T) {
	type In struct {
		Mode string `json:"mode" fugue:"search mode" fugueEnum:"a,b"`
	}
	got := mustReflect(t, In{})
	if !strings.Contains(string(got), `"description":"search mode"`) || !strings.Contains(string(got), `"enum":["a","b"]`) {
		t.Errorf("schema = %s", got)
	}
}

func TestReflect_FugueEnumOnNonStringPanics(t *testing.T) {
	type In struct {
		N int `json:"n" fugueEnum:"1,2,3"`
	}
	_, err := reflectInputSchema(reflect.TypeOf(In{}))
	if err == nil {
		t.Fatal("expected error for fugueEnum on non-string field")
	}
	if !strings.Contains(err.Error(), "fugueEnum") || !strings.Contains(err.Error(), "string") {
		t.Errorf("error should mention fugueEnum on non-string, got: %v", err)
	}
}
